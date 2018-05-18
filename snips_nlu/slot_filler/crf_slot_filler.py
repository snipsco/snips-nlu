from __future__ import unicode_literals

import base64
import io
import logging
import math
import os
import tempfile
from copy import copy
from itertools import groupby, product

from builtins import range
from future.utils import iteritems
from sklearn_crfsuite import CRF

from snips_nlu.builtin_entities import is_builtin_entity, get_builtin_entities
from snips_nlu.constants import (
    RES_MATCH_RANGE, LANGUAGE, DATA, RES_ENTITY, START, END, RES_VALUE,
    ENTITY_KIND, STEMS)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.pipeline.configs import CRFSlotFillerConfig
from snips_nlu.preprocessing import stem
from snips_nlu.resources import resource_exists
from snips_nlu.slot_filler.crf_utils import (
    TOKENS, TAGS, OUTSIDE, tags_to_slots, tag_name_to_slot_name,
    tags_to_preslots, positive_tagging, utterance_to_sample)
from snips_nlu.slot_filler.feature import TOKEN_NAME
from snips_nlu.slot_filler.feature_factory import get_feature_factory
from snips_nlu.slot_filler.slot_filler import SlotFiller
from snips_nlu.tokenization import Token, tokenize
from snips_nlu.utils import (
    UnupdatableDict, mkdir_p, check_random_state, get_slot_name_mapping,
    ranges_overlap, NotTrained, DifferedLoggingMessage, log_elapsed_time)

logger = logging.getLogger(__name__)


class CRFSlotFiller(SlotFiller):
    """Slot filler which uses Linear-Chain Conditional Random Fields underneath

    Check https://en.wikipedia.org/wiki/Conditional_random_field to learn
    more about CRFs
    """

    unit_name = "crf_slot_filler"
    config_type = CRFSlotFillerConfig

    def __init__(self, config=None):
        """The CRF slot filler can be configured by passing a
        :class:`.CRFSlotFillerConfig`"""

        if config is None:
            config = self.config_type()
        super(CRFSlotFiller, self).__init__(config)
        self.crf_model = None
        self.features_factories = [get_feature_factory(conf) for conf in
                                   self.config.feature_factory_configs]
        self._features = None
        self.language = None
        self.intent = None
        self.slot_name_mapping = None

    @property
    def features(self):
        """List of :class:`.Feature` used by the CRF"""
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
        """List of CRF labels

        These labels differ from the slot names as they contain an additional
        prefix which depends on the :class:`.TaggingScheme` that is used
        (BIO by default).
        """
        labels = []
        if self.crf_model.tagger_ is not None:
            labels = [_decode_tag(label) for label in
                      self.crf_model.tagger_.labels()]
        return labels

    @property
    def fitted(self):
        """Whether or not the slot filler has already been fitted"""
        return self.crf_model is not None \
               and self.crf_model.tagger_ is not None

    @log_elapsed_time(logger, logging.DEBUG,
                      "Fitted CRFSlotFiller in {elapsed_time}")
    # pylint:disable=arguments-differ
    def fit(self, dataset, intent):
        """Fit the slot filler

        Args:
            dataset (dict): A valid Snips dataset
            intent (str): The specific intent of the dataset to train
                the slot filler on

        Returns:
            :class:`CRFSlotFiller`: The same instance, trained
        """
        logger.debug("Fitting %s slot filler...", intent)
        dataset = validate_and_format_dataset(dataset)
        self.intent = intent
        self.slot_name_mapping = get_slot_name_mapping(dataset, intent)
        self.language = dataset[LANGUAGE]
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
        # ensure ascii tags
        Y = [[_encode_tag(tag) for tag in sample[TAGS]]
             for sample in crf_samples]
        # pylint: enable=C0103
        self.crf_model = _get_crf_model(self.config.crf_args)
        self.crf_model.fit(X, Y)

        logger.debug(
            "Most relevant features for %s:\n%s", self.intent,
            DifferedLoggingMessage(self.log_weights))
        return self

    # pylint:enable=arguments-differ

    def get_slots(self, text):
        """Extracts slots from the provided text

        Returns:
            list of dict: The list of extracted slots

        Raises:
            NotTrained: When the slot filler is not fitted
        """
        if not self.fitted:
            raise NotTrained("CRFSlotFiller must be fitted")
        tokens = tokenize(text, self.language)
        if not tokens:
            return []
        features = self.compute_features(tokens)
        tags = [_decode_tag(tag) for tag in
                self.crf_model.predict_single(features)]
        slots = tags_to_slots(text, tokens, tags, self.config.tagging_scheme,
                              self.slot_name_mapping)

        builtin_slots_names = set(slot_name for (slot_name, entity) in
                                  iteritems(self.slot_name_mapping)
                                  if is_builtin_entity(entity))
        if not builtin_slots_names:
            return slots

        # Replace tags corresponding to builtin entities by outside tags
        tags = _replace_builtin_tags(tags, builtin_slots_names)
        return self._augment_slots(text, tokens, tags, builtin_slots_names)

    def compute_features(self, tokens, drop_out=False):
        """Compute features on the provided tokens

        The *drop_out* parameters allows to activate drop out on features that
        have a positive drop out ratio. This should only be used during
        training.
        """

        if resource_exists(self.language, STEMS):
            tokens = [
                Token(t.value, t.start, t.end,
                      stem=stem(t.normalized_value, self.language))
                for t in tokens]
        else:
            tokens = [Token(t.value, t.start, t.end, stem=t.normalized_value)
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

    def get_sequence_probability(self, tokens, labels):
        """Gives the joint probability of a sequence of tokens and CRF labels

        Args:
            tokens (list of :class:`.Token`): list of tokens
            labels (list of str): CRF labels with their tagging scheme prefix
                ("B-color", "I-color", "O", etc)

        Note:
            The absolute value returned here is generally not very useful,
            however it can be used to compare a sequence of labels relatively
            to another one.
        """
        features = self.compute_features(tokens)
        return self._get_sequence_probability(features, labels)

    def _get_sequence_probability(self, features, labels):
        if not self.fitted:
            raise NotTrained("CRFSlotFiller must be fitted")

        # Use a default substitution label when a label was not seen during
        # training
        substitution_label = OUTSIDE if OUTSIDE in self.labels else \
            self.labels[0]
        cleaned_labels = [
            _encode_tag(substitution_label if l not in self.labels else l)
            for l in labels]
        self.crf_model.tagger_.set(features)
        return self.crf_model.tagger_.probability(cleaned_labels)

    def log_weights(self):
        """Return a logs for both the label-to-label and label-to-features
         weights"""
        log = ""
        transition_features = self.crf_model.transition_features_
        transition_features = sorted(
            iteritems(transition_features),
            key=lambda transition_weight: math.fabs(transition_weight[1]),
            reverse=True)
        log += "\nTransition weights: \n\n"
        for (state_1, state_2), weight in transition_features:
            log += "\n%s %s: %s" % (
                _decode_tag(state_1), _decode_tag(state_2), weight)
        feature_weights = self.crf_model.state_features_
        feature_weights = sorted(
            iteritems(feature_weights),
            key=lambda feature_weight: math.fabs(feature_weight[1]),
            reverse=True)
        log += "\n\nFeature weights: \n\n"
        for (feat, tag), weight in feature_weights:
            log += "\n%s %s: %s" % (feat, _decode_tag(tag), weight)
        return log

    def _augment_slots(self, text, tokens, tags, builtin_slots_names):
        scope = set(self.slot_name_mapping[slot]
                    for slot in builtin_slots_names)
        builtin_entities = [be for entity_kind in scope
                            for be in get_builtin_entities(text, self.language,
                                                           [entity_kind])]
        # We remove builtin entities which conflicts with custom slots
        # extracted by the CRF
        builtin_entities = _filter_overlapping_builtins(
            builtin_entities, tokens, tags, self.config.tagging_scheme)

        # We resolve conflicts between builtin entities by keeping the longest
        # matches. In case when two builtin entities span the same range, we
        # keep both.
        builtin_entities = _disambiguate_builtin_entities(builtin_entities)

        # We group builtin entities based on their position
        grouped_entities = (
            list(bes)
            for _, bes in groupby(builtin_entities,
                                  key=lambda s: s[RES_MATCH_RANGE][START]))
        grouped_entities = sorted(
            grouped_entities,
            key=lambda entities: entities[0][RES_MATCH_RANGE][START])

        features = self.compute_features(tokens)
        spans_ranges = [entities[0][RES_MATCH_RANGE]
                        for entities in grouped_entities]
        tokens_indexes = _spans_to_tokens_indexes(spans_ranges, tokens)

        # We loop on all possible slots permutations and use the CRF to find
        # the best one in terms of probability
        slots_permutations = _get_slots_permutations(
            grouped_entities, self.slot_name_mapping)
        best_updated_tags = tags
        best_permutation_score = -1
        for slots in slots_permutations:
            updated_tags = copy(tags)
            for slot_index, slot in enumerate(slots):
                indexes = tokens_indexes[slot_index]
                sub_tags_sequence = positive_tagging(
                    self.config.tagging_scheme, slot, len(indexes))
                updated_tags[indexes[0]:indexes[-1] + 1] = sub_tags_sequence
            score = self._get_sequence_probability(features, updated_tags)
            if score > best_permutation_score:
                best_updated_tags = updated_tags
                best_permutation_score = score
        slots = tags_to_slots(text, tokens, best_updated_tags,
                              self.config.tagging_scheme,
                              self.slot_name_mapping)

        return _reconciliate_builtin_slots(text, slots, builtin_entities)

    def to_dict(self):
        """Returns a json-serializable dict"""
        crf_model_data = None

        if self.crf_model is not None:
            crf_model_data = _serialize_crf_model(self.crf_model)

        return {
            "unit_name": self.unit_name,
            "language_code": self.language,
            "intent": self.intent,
            "slot_name_mapping": self.slot_name_mapping,
            "crf_model_data": crf_model_data,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, unit_dict):
        """Creates a :class:`CRFSlotFiller` instance from a dict

        The dict must have been generated with :func:`~CRFSlotFiller.to_dict`
        """
        slot_filler_config = cls.config_type.from_dict(unit_dict["config"])
        slot_filler = cls(config=slot_filler_config)

        crf_model_data = unit_dict["crf_model_data"]
        if crf_model_data is not None:
            crf = _deserialize_crf_model(crf_model_data)
            slot_filler.crf_model = crf
        slot_filler.language = unit_dict["language_code"]
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


def _get_crf_model(crf_args):
    model_filename = crf_args.get("model_filename", None)
    if model_filename is not None:
        directory = os.path.dirname(model_filename)
        if not os.path.isdir(directory):
            mkdir_p(directory)

    return CRF(model_filename=model_filename, **crf_args)


def _replace_builtin_tags(tags, builtin_slot_names):
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


def _filter_overlapping_builtins(builtin_entities, tokens, tags,
                                 tagging_scheme):
    slots = tags_to_preslots(tokens, tags, tagging_scheme)
    ents = []
    for ent in builtin_entities:
        if any(ranges_overlap(ent[RES_MATCH_RANGE], s[RES_MATCH_RANGE])
               for s in slots):
            continue
        ents.append(ent)
    return ents


def _spans_to_tokens_indexes(spans, tokens):
    tokens_indexes = []
    for span in spans:
        indexes = []
        for i, token in enumerate(tokens):
            if span[END] > token.start and span[START] < token.end:
                indexes.append(i)
        tokens_indexes.append(indexes)
    return tokens_indexes


def _reconciliate_builtin_slots(text, slots, builtin_entities):
    for slot in slots:
        if not is_builtin_entity(slot[RES_ENTITY]):
            continue
        for be in builtin_entities:
            if be[ENTITY_KIND] != slot[RES_ENTITY]:
                continue
            be_start = be[RES_MATCH_RANGE][START]
            be_end = be[RES_MATCH_RANGE][END]
            be_length = be_end - be_start
            slot_start = slot[RES_MATCH_RANGE][START]
            slot_end = slot[RES_MATCH_RANGE][END]
            slot_length = slot_end - slot_start
            if be_start <= slot_start and be_end >= slot_end \
                    and be_length > slot_length:
                slot[RES_MATCH_RANGE] = {
                    START: be_start,
                    END: be_end
                }
                slot[RES_VALUE] = text[be_start: be_end]
                break
    return slots


def _disambiguate_builtin_entities(builtin_entities):
    if not builtin_entities:
        return []
    builtin_entities = sorted(
        builtin_entities,
        key=lambda be: be[RES_MATCH_RANGE][END] - be[RES_MATCH_RANGE][START],
        reverse=True)

    disambiguated_entities = [builtin_entities[0]]
    for entity in builtin_entities[1:]:
        entity_rng = entity[RES_MATCH_RANGE]
        conflict = False
        for disambiguated_entity in disambiguated_entities:
            disambiguated_entity_rng = disambiguated_entity[RES_MATCH_RANGE]
            if ranges_overlap(entity_rng, disambiguated_entity_rng):
                conflict = True
                if entity_rng == disambiguated_entity_rng:
                    disambiguated_entities.append(entity)
                break
        if not conflict:
            disambiguated_entities.append(entity)

    return sorted(disambiguated_entities,
                  key=lambda be: be[RES_MATCH_RANGE][START])


def _get_slots_permutations(grouped_entities, slot_name_mapping):
    # We associate to each group of entities the list of slot names that
    # could correspond
    possible_slots = [
        list(set(slot_name for slot_name, ent in iteritems(slot_name_mapping)
                 for entity in entities if ent == entity[ENTITY_KIND]))
        + [OUTSIDE]
        for entities in grouped_entities]
    return product(*possible_slots)


def _encode_tag(tag):
    return base64.b64encode(tag.encode("utf8"))


def _decode_tag(tag):
    return base64.b64decode(tag).decode("utf8")


def _serialize_crf_model(crf_model):
    with io.open(crf_model.modelfile.name, mode='rb') as f:
        crfsuite_data = base64.b64encode(f.read()).decode('ascii')
    return crfsuite_data


def _deserialize_crf_model(crf_model_data):
    b64_data = base64.b64decode(crf_model_data)
    with tempfile.NamedTemporaryFile(suffix=".crfsuite", prefix="model",
                                     delete=False) as f:
        f.write(b64_data)
        f.flush()
        crf = CRF(model_filename=f.name)
    return crf
