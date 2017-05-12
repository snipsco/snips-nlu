from __future__ import unicode_literals

import io
import json
import os
from copy import copy
from itertools import groupby, permutations

from snips_nlu.built_in_entities import BuiltInEntity, get_builtin_entities
from snips_nlu.constants import (DATA, INTENTS, CUSTOM_ENGINE, ENTITY,
                                 MATCH_RANGE)
from snips_nlu.dataset import filter_dataset
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_tagger import CRFTagger
from snips_nlu.slot_filler.crf_utils import (tags_to_slots,
                                             utterance_to_sample,
                                             positive_tagging)
from snips_nlu.slot_filler.data_augmentation import augment_utterances
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import (namedtuple_with_defaults, mkdir_p)

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


class ProbabilisticIntentParser:
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

        tokens = tokenize(text)
        if len(tokens) == 0:
            return []

        tagger = self.crf_taggers[intent]
        tags = tagger.get_tags(tokens)
        intent_slots_mapping = self.slot_name_to_entity_mapping[intent]
        slots = tags_to_slots(text, tokens, tags, tagger.tagging_scheme,
                              intent_slots_mapping)

        # Remove slots corresponding to builtin entities
        slots = [s for s in slots if intent_slots_mapping[s.slot_name] not in
                 BuiltInEntity.built_in_entity_by_label]

        builtin_slots = set(s for s in intent_slots_mapping
                            if intent_slots_mapping[s] in
                            BuiltInEntity.built_in_entity_by_label)
        if len(builtin_slots) == 0:
            return slots

        scope = [BuiltInEntity.from_label(intent_slots_mapping[slot])
                 for slot in builtin_slots]
        builtin_entities = get_builtin_entities(text, self.language, scope)
        slots = augment_slots(text, tokens, tags, tagger, intent_slots_mapping,
                              builtin_entities, builtin_slots)
        return slots

    @property
    def fitted(self):
        return self.intent_classifier.fitted and all(
            slot_filler.fitted for slot_filler in self.crf_taggers.values())

    def fit(self, dataset):
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE)
        self.intent_classifier = self.intent_classifier.fit(dataset)
        for intent_name in custom_dataset[INTENTS]:
            augmented_intent_utterances = augment_utterances(
                dataset, intent_name, language=self.language,
                **self.data_augmentation_config.to_dict())
            tagging_scheme = self.crf_taggers[intent_name].tagging_scheme
            crf_samples = [utterance_to_sample(u[DATA], tagging_scheme)
                           for u in augmented_intent_utterances]
            self.crf_taggers[intent_name] = self.crf_taggers[intent_name].fit(
                crf_samples)
        return self

    def save(self, directory_path):
        if not os.path.isdir(directory_path):
            mkdir_p(directory_path)

        parser_config = {
            "language_code": self.language.iso_code,
            "intent_classifier": self.intent_classifier.to_dict(),
            "slot_name_to_entity_mapping": self.slot_name_to_entity_mapping,
            "data_augmentation_config": self.data_augmentation_config.to_dict()
        }
        config_path = os.path.join(directory_path,
                                   "probabilistic_parser_config.json")

        with io.open(config_path, mode='w') as f:
            json_config = json.dumps(parser_config, indent=4).decode(
                encoding='utf8')
            f.write(json_config)

        taggers_directory = os.path.join(directory_path, "taggers")
        if not os.path.isdir(taggers_directory):
            mkdir_p(taggers_directory)
        for intent, tagger in self.crf_taggers.iteritems():
            tagger_directory = os.path.join(taggers_directory, intent)
            tagger.save(tagger_directory)

    @classmethod
    def load(cls, directory_path):
        config_path = os.path.join(directory_path,
                                   "probabilistic_parser_config.json")

        with io.open(config_path) as f:
            parser_config = json.load(f)

        taggers = dict()
        taggers_directory = os.path.join(directory_path, "taggers")
        for intent in os.listdir(taggers_directory):
            tagger_directory = os.path.join(taggers_directory, intent)
            taggers[intent] = CRFTagger.load(tagger_directory)

        return cls(
            language=Language.from_iso_code(parser_config["language_code"]),
            intent_classifier=SnipsIntentClassifier.from_dict(
                parser_config["intent_classifier"]),
            crf_taggers=taggers,
            slot_name_to_entity_mapping=parser_config[
                "slot_name_to_entity_mapping"],
            data_augmentation_config=DataAugmentationConfig.from_dict(
                parser_config["data_augmentation_config"])
        )


def augment_slots(text, tokens, tags, tagger, intent_slots_mapping,
                  builtin_entities, missing_slots):
    augmented_tags = tags
    grouped_entities = groupby(builtin_entities, key=lambda s: s[ENTITY])
    for entity, matches in grouped_entities:
        spans_ranges = [match[MATCH_RANGE] for match in matches]
        tokens_indexes = spans_to_tokens_indexes(spans_ranges, tokens)
        related_slots = set(s for s in missing_slots
                            if intent_slots_mapping[s] == entity.label)
        slots_permutations = permutations(related_slots)
        best_updated_tags = augmented_tags
        best_permutation_score = -1
        for slots in slots_permutations:
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
