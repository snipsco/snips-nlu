from __future__ import unicode_literals
from __future__ import print_function

import base64
import io
import math
import os
import tempfile
from copy import copy
from itertools import groupby, permutations, product

from sklearn_crfsuite import CRF

from snips_nlu.builtin_entities import is_builtin_entity, BuiltInEntity, \
    get_builtin_entities
from snips_nlu.constants import RES_MATCH_RANGE, ENTITY, LANGUAGE, DATA
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.languages import Language
from snips_nlu.pipeline.configs.slot_filler import CRFSlotFillerConfig
from snips_nlu.preprocessing import stem
from snips_nlu.slot_filler.crf_utils import TOKENS, TAGS, OUTSIDE, \
    tags_to_slots, tag_name_to_slot_name, tags_to_preslots, positive_tagging, \
    utterance_to_sample
from snips_nlu.slot_filler.feature import TOKEN_NAME
from snips_nlu.slot_filler.feature_factory import get_feature_factory
from snips_nlu.slot_filler.slot_filler import SlotFiller
from snips_nlu.tokenization import Token, tokenize
from snips_nlu.utils import UnupdatableDict, mkdir_p, check_random_state, \
    get_slot_name_mapping, ranges_overlap


class CRFSlotFiller(SlotFiller):
    unit_name = "crf_slot_filler"
    config_type = CRFSlotFillerConfig

    def __init__(self, config=None):
        if config is None:
            config = self.config_type()
        super(CRFSlotFiller, self).__init__(config)
        self.crf_model = None
        self.features_factories = [get_feature_factory(conf) for conf in
                                   config.feature_factory_configs]
        self._features = None
        self.language = None
        self.intent = None
        self.slot_name_mapping = None

    @property
    def features(self):
        if self._features is None:
            self._features = []
            feature_names = set()
            for factory in self.features_factories:
                for feature in factory.build_features():
                    if feature.name in feature_names:
                        raise KeyError("Duplicated feature: %s" % feature.name)
                    feature_names.add(feature.name)
                    self._features.append(feature)
        return self._features

    @property
    def labels(self):
        labels = []
        if self.crf_model.tagger_ is not None:
            labels = [label.decode('utf8') for label in
                      self.crf_model.tagger_.labels()]
        return labels

    @property
    def fitted(self):
        return self.crf_model is not None \
               and self.crf_model.tagger_ is not None

    def get_slots(self, text):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")
        tokens = tokenize(text, self.language)
        if not tokens:
            return []
        features = self.compute_features(tokens)
        tags = [tag.decode('utf8') for tag in
                self.crf_model.predict_single(features)]
        slots = tags_to_slots(text, tokens, tags, self.config.tagging_scheme,
                              self.slot_name_mapping)

        builtin_slots_names = set(slot_name for (slot_name, entity) in
                                  self.slot_name_mapping.iteritems()
                                  if is_builtin_entity(entity))
        if not builtin_slots_names:
            return slots

        # Replace tags corresponding to builtin entities by outside tags
        tags = replace_builtin_tags(tags, builtin_slots_names)
        return self._augment_slots(text, tokens, tags, builtin_slots_names)

    def get_sequence_probability(self, features, labels):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")

        # Use a default substitution label when a label was not seen during
        # training
        substitution_label = OUTSIDE if OUTSIDE in self.labels else \
            self.labels[0]
        cleaned_labels = [substitution_label if l not in self.labels else l for
                          l in labels]
        cleaned_labels = [label.encode('utf8') for label in cleaned_labels]
        self.crf_model.tagger_.set(features)
        return self.crf_model.tagger_.probability(cleaned_labels)

    # pylint:disable=arguments-differ
    def fit(self, dataset, intent, verbose=False):
        self.intent = intent
        self.slot_name_mapping = get_slot_name_mapping(dataset, intent)
        self.language = Language.from_iso_code(dataset[LANGUAGE])
        random_state = check_random_state(self.config.random_seed)
        augmented_intent_utterances = augment_utterances(
            dataset, self.intent, language=self.language,
            random_state=random_state,
            **self.config.data_augmentation_config.to_dict())

        crf_samples = [
            utterance_to_sample(u[DATA], self.config.tagging_scheme,
                                self.language)
            for u in augmented_intent_utterances]

        for factory in self.features_factories:
            factory.fit(dataset, intent)

        # pylint: disable=C0103
        X = [self.compute_features(sample[TOKENS], drop_out=True)
             for sample in crf_samples]
        Y = [[tag.encode('utf8') for tag in sample[TAGS]]
             for sample in crf_samples]
        # pylint: enable=C0103
        self.crf_model = get_crf_model(self.config.crf_args)
        self.crf_model.fit(X, Y)
        if verbose:
            self.print_weights()

        return self

    # pylint:enable=arguments-differ

    def print_weights(self):
        transition_features = self.crf_model.transition_features_
        transition_features = sorted(
            transition_features.iteritems(),
            key=lambda transition_weight: math.fabs(transition_weight[1]),
            reverse=True)
        print("\nTransition weights: \n\n")
        for (state_1, state_2), weight in transition_features:
            print("%s %s: %s" % (state_1, state_2, weight))
        feature_weights = self.crf_model.state_features_
        feature_weights = sorted(
            feature_weights.iteritems(),
            key=lambda feature_weight: math.fabs(feature_weight[1]),
            reverse=True)
        print("\nFeature weights: \n\n")
        for (feat, tag), weight in feature_weights:
            print("%s %s: %s" % (feat, tag, weight))

    def compute_features(self, tokens, drop_out=False):
        tokens = [
            Token(t.value, t.start, t.end,
                  stem=stem(t.normalized_value, self.language))
            for t in tokens]
        cache = [{TOKEN_NAME: token} for token in tokens]
        features = []
        random_state = check_random_state(self.config.random_seed)
        for i in range(len(tokens)):
            token_features = UnupdatableDict()
            for feature in self.features:
                f_drop_out = feature.drop_out
                if drop_out and random_state.rand() < f_drop_out:
                    continue
                value = feature.compute(i, cache)
                if value is not None:
                    token_features[feature.name] = value
            features.append(token_features)
        return features

    def _augment_slots(self, text, tokens, tags, builtin_slots_names):
        augmented_tags = tags
        scope = [BuiltInEntity.from_label(self.slot_name_mapping[slot])
                 for slot in builtin_slots_names]
        builtin_entities = get_builtin_entities(text, self.language, scope)

        builtin_entities = filter_overlapping_builtins(
            builtin_entities, tokens, tags, self.config.tagging_scheme)

        grouped_entities = groupby(builtin_entities, key=lambda s: s[ENTITY])
        features = None
        for entity, matches in grouped_entities:
            spans_ranges = [match[RES_MATCH_RANGE] for match in matches]
            num_possible_builtins = len(spans_ranges)
            tokens_indexes = spans_to_tokens_indexes(spans_ranges, tokens)
            related_slots = list(
                set(s for s in builtin_slots_names if
                    self.slot_name_mapping[s] == entity.label))
            best_updated_tags = augmented_tags
            best_permutation_score = -1

            for slots in generate_slots_permutations(
                    num_possible_builtins, related_slots,
                    self.config.exhaustive_permutations_threshold):
                updated_tags = copy(augmented_tags)
                for slot_index, slot in enumerate(slots):
                    if slot_index >= len(tokens_indexes):
                        break
                    indexes = tokens_indexes[slot_index]
                    sub_tags_sequence = positive_tagging(
                        self.config.tagging_scheme, slot, len(indexes))
                    updated_tags[indexes[0]:indexes[-1] + 1] = \
                        sub_tags_sequence
                if features is None:
                    features = self.compute_features(tokens)
                score = self.get_sequence_probability(features, updated_tags)
                if score > best_permutation_score:
                    best_updated_tags = updated_tags
                    best_permutation_score = score
            augmented_tags = best_updated_tags
        return tags_to_slots(text, tokens, augmented_tags,
                             self.config.tagging_scheme,
                             self.slot_name_mapping)

    def to_dict(self):
        language_code = None
        crf_model_data = None

        if self.language is not None:
            language_code = self.language.iso_code
        if self.crf_model is not None:
            crf_model_data = serialize_crf_model(self.crf_model)

        return {
            "unit_name": self.unit_name,
            "language_code": language_code,
            "intent": self.intent,
            "slot_name_mapping": self.slot_name_mapping,
            "crf_model_data": crf_model_data,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, unit_dict):
        slot_filler_config = cls.config_type.from_dict(unit_dict["config"])
        slot_filler = cls(config=slot_filler_config)

        crf_model_data = unit_dict["crf_model_data"]
        if crf_model_data is not None:
            crf = deserialize_crf_model(crf_model_data)
            slot_filler.crf_model = crf
        language_code = unit_dict["language_code"]
        if language_code is not None:
            language = Language.from_iso_code(language_code)
            slot_filler.language = language
        slot_filler.intent = unit_dict["intent"]
        slot_filler.slot_name_mapping = unit_dict["slot_name_mapping"]
        return slot_filler

    def __del__(self):
        if self.crf_model is None or self.crf_model.modelfile.name is None:
            return
        try:
            os.remove(self.crf_model.modelfile.name)
        except OSError:
            pass


def get_crf_model(crf_args):
    model_filename = crf_args.get("model_filename", None)
    if model_filename is not None:
        directory = os.path.dirname(model_filename)
        if not os.path.isdir(directory):
            mkdir_p(directory)

    return CRF(model_filename=model_filename, **crf_args)


def replace_builtin_tags(tags, builtin_slot_names):
    new_tags = []
    for tag in tags:
        if tag == OUTSIDE:
            new_tags.append(tag)
        else:
            slot_name = tag_name_to_slot_name(tag)
            if slot_name in builtin_slot_names:
                new_tags.append(OUTSIDE)
            else:
                new_tags.append(tag)
    return new_tags


def filter_overlapping_builtins(builtin_entities, tokens, tags,
                                tagging_scheme):
    slots = tags_to_preslots(tokens, tags, tagging_scheme)
    ents = []
    for ent in builtin_entities:
        if any(ranges_overlap(ent[RES_MATCH_RANGE], s[RES_MATCH_RANGE])
               for s in slots):
            continue
        ents.append(ent)
    return ents


def generate_slots_permutations(n_detected_builtins, possible_slots_names,
                                exhaustive_permutations_threshold):
    num_exhaustive_perms = (len(possible_slots_names) + 1) \
                           ** n_detected_builtins
    if num_exhaustive_perms <= exhaustive_permutations_threshold:
        return exhaustive_slots_permutations(
            n_detected_builtins, possible_slots_names)
    return conservative_slots_permutations(
        n_detected_builtins, possible_slots_names)


def exhaustive_slots_permutations(n_detected_builtins, possible_slots_names):
    pool = possible_slots_names + [OUTSIDE]
    return [p for p in product(pool, repeat=n_detected_builtins) if len(p)]


def conservative_slots_permutations(n_detected_builtins, possible_slots_names):
    if n_detected_builtins == 0:
        return []
    # Add n_detected_builtins "O" slots to the possible slots.
    # It's possible that out of the detected builtins the CRF choose that
    # none of them are likely to be an actually slot, these combination
    # must be taken into account
    permutation_pool = range(len(possible_slots_names) + n_detected_builtins)
    # Generate all permutations
    perms = permutations(permutation_pool, n_detected_builtins)

    # Replace the indices greater than possible_slots_names by "O"
    perms = [tuple(possible_slots_names[i] if i < len(possible_slots_names)
                   else OUTSIDE for i in p) for p in perms]
    # Make the permutations unique
    return list(set(perms))


def spans_to_tokens_indexes(spans, tokens):
    tokens_indexes = []
    for span_start, span_end in spans:
        indexes = []
        for i, token in enumerate(tokens):
            if span_end > token.start and span_start < token.end:
                indexes.append(i)
        tokens_indexes.append(indexes)
    return tokens_indexes


def serialize_crf_model(crf_model):
    with io.open(crf_model.modelfile.name, mode='rb') as f:
        crfsuite_data = base64.b64encode(f.read()).decode('ascii')
    return crfsuite_data


def deserialize_crf_model(crf_model_data):
    b64_data = base64.b64decode(crf_model_data)
    with tempfile.NamedTemporaryFile(suffix=".crfsuite", prefix="model",
                                     delete=False) as f:
        f.write(b64_data)
        f.flush()
        crf = CRF(model_filename=f.name)
    return crf
