from copy import copy
from itertools import permutations

from intent_parser import IntentParser
from snips_nlu.built_in_entities import get_built_in_entities, BuiltInEntity
from snips_nlu.constants import (DATA, INTENTS, SLOT_NAME, UTTERANCES, ENTITY,
                                 CUSTOM_ENGINE, MATCH_RANGE)
from snips_nlu.dataset import filter_dataset
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot
from snips_nlu.slot_filler.crf_tagger import CRFTagger
from snips_nlu.slot_filler.crf_utils import (tags_to_slots,
                                             utterance_to_sample,
                                             positive_tagging)
from snips_nlu.slot_filler.data_augmentation import augment_utterances
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import (instance_to_generic_dict, instance_from_dict,
                             namedtuple_with_defaults)

from itertools import groupby

_DataAugmentationConfig = namedtuple_with_defaults(
    '_DataAugmentationConfig',
    'max_utterances noise_prob min_noise_size max_noise_size',
    {
        'max_utterances': 0,
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


def default_data_augmentation_config(language):
    if language == Language.EN:
        return DataAugmentationConfig(max_utterances=200, noise_prob=0.05,
                                      min_noise_size=1, max_noise_size=3)
    else:
        return DataAugmentationConfig()


def get_slot_name_to_entity_mapping(dataset):
    slot_name_to_entity = dict()
    for intent_name, intent in dataset[INTENTS].iteritems():
        _dict = dict()
        slot_name_to_entity[intent_name] = _dict
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    _dict[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_to_entity


class CRFIntentParser(IntentParser):
    def __init__(self, language, intent_classifier, crf_taggers,
                 slot_name_to_entity_mapping=None,
                 data_augmentation_config=None):
        super(CRFIntentParser, self).__init__()
        self.language = language
        self.intent_classifier = intent_classifier
        self._crf_taggers = None
        self.crf_taggers = crf_taggers
        self.slot_name_to_entity_mapping = slot_name_to_entity_mapping
        if data_augmentation_config is None:
            data_augmentation_config = default_data_augmentation_config(
                self.language)
        self.data_augmentation_config = data_augmentation_config
        self.intents_data_sizes = {intent: 0 for intent in self.crf_taggers}

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
            raise ValueError("CRFIntentParser must be fitted before "
                             "`get_intent` is called")
        return self.intent_classifier.get_intent(text)

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        if not self.fitted:
            raise ValueError("CRFIntentParser must be fitted before "
                             "`get_slots` is called")
        if intent not in self.crf_taggers:
            raise KeyError("Invalid intent '%s'" % intent)

        tokens = tokenize(text)
        tagger = self.crf_taggers[intent]
        tags = tagger.get_tags(tokens)
        slots = tags_to_slots(tokens, tags, tagger.tagging_scheme)
        intent_slots_mapping = self.slot_name_to_entity_mapping[intent]
        parsed_slots = [
            ParsedSlot(match_range=slot["range"],
                       value=text[slot["range"][0]:slot["range"][1]],
                       entity=intent_slots_mapping[slot[SLOT_NAME]],
                       slot_name=slot[SLOT_NAME])
            for slot in slots
        ]
        all_intent_slots = intent_slots_mapping.keys()
        found_slots = set(s[SLOT_NAME] for s in slots)
        missing_slots = set(all_intent_slots).difference(found_slots)
        missing_builtin_slots = [slot for slot in missing_slots
                                 if intent_slots_mapping[slot] in
                                 BuiltInEntity.built_in_entity_by_label]

        if self.intents_data_sizes[intent] <= 20 \
                and len(missing_builtin_slots) > 0:
            scope = set(BuiltInEntity.from_label(intent_slots_mapping[slot])
                        for slot in missing_builtin_slots)
            builtin_entities = get_built_in_entities(text, self.language,
                                                     list(scope))
            parsed_slots = _augment_slots(text, intent_slots_mapping,
                                          builtin_entities, missing_slots,
                                          tokens, tags, tagger)

        return parsed_slots

    @property
    def fitted(self):
        return self.intent_classifier.fitted and all(
            slot_filler.fitted for slot_filler in self.crf_taggers.values())

    def fit(self, dataset):
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE)
        self.intents_data_sizes = {intent_name: len(intent[UTTERANCES])
                                   for intent_name, intent
                                   in custom_dataset[INTENTS].iteritems()}
        self.slot_name_to_entity_mapping = get_slot_name_to_entity_mapping(
            custom_dataset)
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

    def to_dict(self):
        obj_dict = instance_to_generic_dict(self)
        obj_dict.update({
            "language_code": self.language.iso_code,
            "intent_classifier": self.intent_classifier.to_dict(),
            "crf_taggers": {intent_name: tagger.to_dict() for
                            intent_name, tagger in
                            self.crf_taggers.iteritems()},
            "slot_name_to_entity_mapping": self.slot_name_to_entity_mapping,
            "data_augmentation_config": self.data_augmentation_config.to_dict()
        })
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(
            language=Language.from_iso_code(obj_dict["language_code"]),
            intent_classifier=instance_from_dict(
                obj_dict["intent_classifier"]),
            crf_taggers={intent_name: CRFTagger.from_dict(tagger_dict)
                         for intent_name, tagger_dict in
                         obj_dict["crf_taggers"].iteritems()},
            slot_name_to_entity_mapping=obj_dict[
                "slot_name_to_entity_mapping"],
            data_augmentation_config=DataAugmentationConfig.from_dict(
                obj_dict["data_augmentation_config"])
        )


def _augment_slots(text, intent_slots_mapping, builtin_slots, missing_slots,
                   tokens, tags, tagger):
    augmented_tags = copy(tags)
    grouped_entities = groupby(builtin_slots, key=lambda s: s[ENTITY])
    for entity, spans in grouped_entities:
        spans_ranges = [span[MATCH_RANGE] for span in spans]
        token_indexes = _spans_to_token_indexes(spans_ranges, tokens)
        related_slots = set(s for s in missing_slots
                            if intent_slots_mapping[s] == entity.label)
        slots_permutations = permutations(related_slots)
        best_permutation_tags = None
        best_permutation_score = None
        for slots in slots_permutations:
            updated_tags = copy(augmented_tags)
            for slot_index, slot in enumerate(slots):
                if slot_index >= len(token_indexes):
                    break
                _token_indexes = token_indexes[slot_index]
                _tags = positive_tagging(tagger.tagging_scheme, slot,
                                         len(token_indexes))
                updated_tags[_token_indexes[0]:_token_indexes[-1] + 1] = _tags
            score = tagger.get_sequence_probability(tokens, updated_tags)
            if best_permutation_score is None \
                    or score > best_permutation_score:
                best_permutation_tags = updated_tags
                best_permutation_score = score
        augmented_tags = best_permutation_tags
    augmented_slots = tags_to_slots(tokens, augmented_tags,
                                    tagger.tagging_scheme)
    return [
        ParsedSlot(match_range=slot["range"],
                   value=text[slot["range"][0]:slot["range"][1]],
                   entity=intent_slots_mapping[slot[SLOT_NAME]],
                   slot_name=slot[SLOT_NAME])
        for slot in augmented_slots
    ]


def _spans_to_token_indexes(spans, tokens):
    token_indexes = []
    for span_start, span_end in spans:
        indexes = []
        for i, token in enumerate(tokens):
            if span_end > token.start and span_start < token.end:
                indexes.append(i)
        token_indexes.append(indexes)
    return token_indexes
