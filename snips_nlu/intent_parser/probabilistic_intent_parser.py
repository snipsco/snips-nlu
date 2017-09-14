from __future__ import unicode_literals

import random
from copy import copy, deepcopy
from itertools import groupby, permutations

from snips_nlu.builtin_entities import BuiltInEntity, get_builtin_entities, \
    is_builtin_entity
from snips_nlu.constants import (DATA, INTENTS, ENTITY,
                                 MATCH_RANGE, ENTITIES, CAPITALIZE, TEXT)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.languages import Language
from snips_nlu.resources import get_stop_words
from snips_nlu.slot_filler.crf_tagger import CRFTagger
from snips_nlu.slot_filler.crf_utils import (tags_to_slots,
                                             utterance_to_sample,
                                             positive_tagging, OUTSIDE,
                                             tag_name_to_slot_name)
from snips_nlu.tokenization import tokenize, tokenize_light
from snips_nlu.utils import (namedtuple_with_defaults)

_DataAugmentationConfig = namedtuple_with_defaults(
    '_DataAugmentationConfig',
    'max_utterances noise_prob min_noise_size max_noise_size',
    {
        'max_utterances': 200,
        'noise_prob': 0.,
        'min_noise_size': 0,
        'max_noise_size': 0
    }
)


class DataAugmentationConfig(_DataAugmentationConfig):
    def to_dict(self):
        return self._asdict()

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


def capitalize(text, language):
    tokens = tokenize_light(text, language)
    return language.default_sep.join(
        t.title() if t.lower() not in get_stop_words(language)
        else t.lower() for t in tokens)


def capitalize_utterances(utterances, entities, language, ratio=.2):
    # TODO: put it in a capitalization config in the probabilistic parser
    # but it breaks serialization -> wait for it
    capitalized_utterances = []
    for utterance in utterances:
        capitalized_utterance = deepcopy(utterance)
        for i, chunk in enumerate(capitalized_utterance[DATA]):
            if ENTITY not in chunk:
                continue
            entity_label = chunk[ENTITY]
            if is_builtin_entity(entity_label):
                continue
            if not entities[entity_label][CAPITALIZE]:
                continue
            if random.random() > ratio:
                continue
            capitalized_utterance[DATA][i][TEXT] = capitalize(
                chunk[TEXT], language)
        capitalized_utterances.append(capitalized_utterance)
    return capitalized_utterances


def fit_tagger(tagger, dataset, intent_name, language,
               data_augmentation_config):
    augmented_intent_utterances = augment_utterances(
        dataset, intent_name, language=language,
        **data_augmentation_config.to_dict())
    augmented_intent_utterances = capitalize_utterances(
        augmented_intent_utterances, dataset[ENTITIES], language)
    tagging_scheme = tagger.tagging_scheme
    crf_samples = [
        utterance_to_sample(u[DATA], tagging_scheme, tagger.language)
        for u in augmented_intent_utterances]
    return tagger.fit(crf_samples)


class ProbabilisticIntentParser(object):
    def __init__(self, language, intent_classifier, crf_taggers,
                 slot_name_to_entity_mapping, data_augmentation_config=None):
        self.language = language
        self.intent_classifier = intent_classifier
        self._crf_taggers = None
        self.crf_taggers = crf_taggers
        self.slot_name_to_entity_mapping = slot_name_to_entity_mapping
        if data_augmentation_config is None:
            data_augmentation_config = DataAugmentationConfig()
        self.data_augmentation_config = data_augmentation_config

    @property
    def crf_taggers(self):
        return self._crf_taggers

    @crf_taggers.setter
    def crf_taggers(self, value):
        if any(t.language != self.language for t in value.values()):
            raise ValueError("Found taggers with different languages")
        self._crf_taggers = value

    def get_intent(self, text):
        if not self.fitted:
            raise ValueError("ProbabilisticIntentParser must be fitted before "
                             "`get_intent` is called")
        return self.intent_classifier.get_intent(text)

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        if not self.fitted:
            raise ValueError("ProbabilisticIntentParser must be fitted before "
                             "`get_slots` is called")
        if intent not in self.crf_taggers:
            raise KeyError("Invalid intent '%s'" % intent)

        tokens = tokenize(text, self.language)
        if len(tokens) == 0:
            return []

        tagger = self.crf_taggers[intent]
        tags = tagger.get_tags(tokens)
        intent_slots_mapping = self.slot_name_to_entity_mapping[intent]
        slots = tags_to_slots(text, tokens, tags, tagger.tagging_scheme,
                              intent_slots_mapping)

        builtin_slot_names = set(slot_name for (slot_name, entity) in
                                 intent_slots_mapping.iteritems() if entity
                                 in BuiltInEntity.built_in_entity_by_label)
        if len(builtin_slot_names) == 0:
            return slots

        # Replace tags corresponding to builtin entities by outside tags
        tags = replace_builtin_tags(tags, builtin_slot_names)

        scope = [BuiltInEntity.from_label(intent_slots_mapping[slot])
                 for slot in builtin_slot_names]
        builtin_entities = get_builtin_entities(text, self.language, scope)
        slots = augment_slots(text, tokens, tags, tagger, intent_slots_mapping,
                              builtin_entities, builtin_slot_names)
        return slots

    @property
    def fitted(self):
        return self.intent_classifier.fitted and all(
            slot_filler.fitted for slot_filler in self.crf_taggers.values())

    def fit(self, dataset, intents=None):
        if intents is None:
            intents = set(dataset[INTENTS].keys())
        self.intent_classifier = self.intent_classifier.fit(dataset)
        for intent_name in dataset[INTENTS]:
            if intent_name not in intents:
                continue
            self.crf_taggers[intent_name] = fit_tagger(
                self.crf_taggers[intent_name], dataset, intent_name,
                self.language, self.data_augmentation_config)
        return self

    def to_dict(self):
        taggers = {intent: tagger.to_dict()
                   for intent, tagger in self.crf_taggers.iteritems()}

        return {
            "language_code": self.language.iso_code,
            "intent_classifier": self.intent_classifier.to_dict(),
            "slot_name_to_entity_mapping": self.slot_name_to_entity_mapping,
            "data_augmentation_config":
                self.data_augmentation_config.to_dict(),
            "taggers": taggers
        }

    @classmethod
    def from_dict(cls, obj_dict):
        taggers = {intent: CRFTagger.from_dict(tagger_dict) for
                   intent, tagger_dict in obj_dict["taggers"].iteritems()}

        return cls(
            language=Language.from_iso_code(obj_dict["language_code"]),
            intent_classifier=SnipsIntentClassifier.from_dict(
                obj_dict["intent_classifier"]),
            crf_taggers=taggers,
            slot_name_to_entity_mapping=obj_dict[
                "slot_name_to_entity_mapping"],
            data_augmentation_config=DataAugmentationConfig.from_dict(
                obj_dict["data_augmentation_config"])
        )


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


def generate_slots_permutations(n_detected_builtins, possible_slots_names):
    if n_detected_builtins == 0:
        return []
    # Add n_detected_builtins "O" slots to the possible slots.
    # It's possible that out of the detected builtins the CRF choose that
    # none of them are likely to be an actually slot, these combination
    # must be taken into account
    permutation_pool = range(len(possible_slots_names) + n_detected_builtins)

    # Generate all permutations
    perms = [p for p in permutations(permutation_pool, n_detected_builtins)]

    # Replace the indices greater than possible_slots_names by "O"
    perms = [tuple(possible_slots_names[i] if i < len(possible_slots_names)
                   else OUTSIDE for i in p) for p in perms]

    # Make the permutations unique
    return list(set(perms))


def augment_slots(text, tokens, tags, tagger, intent_slots_mapping,
                  builtin_entities, builtin_slots_names):
    augmented_tags = tags
    grouped_entities = groupby(builtin_entities, key=lambda s: s[ENTITY])
    for entity, matches in grouped_entities:
        spans_ranges = [match[MATCH_RANGE] for match in matches]
        num_possible_builtins = len(spans_ranges)
        tokens_indexes = spans_to_tokens_indexes(spans_ranges, tokens)
        related_slots = list(set(s for s in builtin_slots_names
                                 if intent_slots_mapping[s] == entity.label))
        best_updated_tags = augmented_tags
        best_permutation_score = -1
        for slots in generate_slots_permutations(num_possible_builtins,
                                                 related_slots):
            updated_tags = copy(augmented_tags)
            for slot_index, slot in enumerate(slots):
                if slot_index >= len(tokens_indexes):
                    break
                indexes = tokens_indexes[slot_index]
                sub_tags_sequence = positive_tagging(tagger.tagging_scheme,
                                                     slot, len(indexes))
                updated_tags[indexes[0]:indexes[-1] + 1] = sub_tags_sequence
            score = tagger.get_sequence_probability(tokens, updated_tags)
            if score > best_permutation_score:
                best_updated_tags = updated_tags
                best_permutation_score = score
        augmented_tags = best_updated_tags
    return tags_to_slots(text, tokens, augmented_tags, tagger.tagging_scheme,
                         intent_slots_mapping)


def spans_to_tokens_indexes(spans, tokens):
    tokens_indexes = []
    for span_start, span_end in spans:
        indexes = []
        for i, token in enumerate(tokens):
            if span_end > token.start and span_start < token.end:
                indexes.append(i)
        tokens_indexes.append(indexes)
    return tokens_indexes
