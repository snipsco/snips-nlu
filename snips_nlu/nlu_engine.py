from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from copy import copy
from itertools import groupby, permutations

from dataset import validate_and_format_dataset, filter_dataset
from snips_nlu.built_in_entities import BuiltInEntity, get_built_in_entities, \
    is_builtin_entity
from snips_nlu.constants import (
    INTENTS, ENTITIES, UTTERANCES, LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE,
    ENTITY, BUILTIN_PARSER, CUSTOM_ENGINE, MATCH_RANGE, DATA, SLOT_NAME,
    USE_SYNONYMS, SYNONYMS)
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.builtin_intent_parser import BuiltinIntentParser
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot, empty_result, \
    IntentClassificationResult
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme, positive_tagging, \
    tags_to_slots
from snips_nlu.slot_filler.feature_functions import crf_features
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import instance_from_dict


class NLUEngine(object):
    __metaclass__ = ABCMeta

    def __init__(self, language):
        self._language = None
        self.language = language

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if isinstance(value, Language):
            self._language = value
        elif isinstance(value, (str, unicode)):
            self._language = Language.from_iso_code(value)
        else:
            raise TypeError("Expected str, unicode or Language found '%s'"
                            % type(value))

    @abstractmethod
    def parse(self, text):
        pass


def _parse(text, entities, rule_based_parser=None, probabilistic_parser=None,
           builtin_parser=None, intent=None):
    parsers = []
    if rule_based_parser is not None:
        parsers.append(rule_based_parser)
    if probabilistic_parser is not None:
        parsers.append(probabilistic_parser)

    if intent is None and builtin_parser is not None:  # if the intent is given
        #  it's a custom intent
        parsers.append(builtin_parser)
    if len(parsers) == 0:
        return empty_result(text)

    for parser in parsers:
        if intent is None:
            res = parser.get_intent(text)
            if res is None:
                continue
            intent_name = res.intent_name
        else:
            res = IntentClassificationResult(intent, 1.0)
            intent_name = intent
        valid_slot = []
        slots = parser.get_slots(text, intent_name)
        for s in slots:
            slot_value = s.value
            # Check if the entity is from a custom intent
            if s.entity in entities:
                entity = entities[s.entity]
                if not entity[AUTOMATICALLY_EXTENSIBLE]:
                    if s.value not in entity[UTTERANCES]:
                        continue
                    slot_value = entity[UTTERANCES][s.value]
            s = ParsedSlot(s.match_range, slot_value, s.entity,
                           s.slot_name)
            valid_slot.append(s)
        return Result(text, parsed_intent=res, parsed_slots=valid_slot)
    return empty_result(text)


def augment_slots(text, tagger, intent_slots_mapping, builtin_entities,
                  missing_slots):
    tokens = tokenize(text)
    # TODO: Find a way to avoid tagging multiple times
    tags = tagger.get_tags(tokens)
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


def get_slot_name_mapping(dataset):
    """
    Returns a dict which maps slot names to entities
    """
    slot_name_mapping = dict()
    for intent_name, intent in dataset[INTENTS].iteritems():
        _dict = dict()
        slot_name_mapping[intent_name] = _dict
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    _dict[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def get_intent_custom_entities(dataset, intent):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent][UTTERANCES]:
        for c in utterance[DATA]:
            if ENTITY in c:
                intent_entities.add(c[ENTITY])
    custom_entities = dict()
    for ent in intent_entities:
        if ent not in BuiltInEntity.built_in_entity_by_label:
            custom_entities[ent] = dataset[ENTITIES][ent]
    return custom_entities


def snips_nlu_entities(dataset):
    entities = dict()
    for entity_name, entity in dataset[ENTITIES].iteritems():
        if is_builtin_entity(entity_name):
            continue
        entity_data = dict()
        use_synonyms = entity[USE_SYNONYMS]
        automatically_extensible = entity[AUTOMATICALLY_EXTENSIBLE]
        entity_data[AUTOMATICALLY_EXTENSIBLE] = automatically_extensible

        entity_utterances = dict()
        for data in entity[DATA]:
            if use_synonyms:
                for s in data[SYNONYMS]:
                    entity_utterances[s] = data[VALUE]
            else:
                entity_utterances[data[VALUE]] = data[VALUE]
        entity_data[UTTERANCES] = entity_utterances
        entities[entity_name] = entity_data
    return entities


class SnipsNLUEngine(NLUEngine):
    def __init__(self, language, rule_based_parser=None,
                 probabilistic_parser=None, builtin_parser=None, entities=None,
                 slot_name_mapping=None, ui_builtin_parsing_threshold=None,
                 intents_data_sizes=None):
        super(SnipsNLUEngine, self).__init__(language)
        self.rule_based_parser = rule_based_parser
        self.probabilistic_parser = probabilistic_parser
        self._builtin_parser = None
        self.builtin_parser = builtin_parser
        if entities is None:
            entities = dict()
        self.entities = entities
        if slot_name_mapping is None:
            slot_name_mapping = dict()
        self.slot_name_mapping = slot_name_mapping
        if ui_builtin_parsing_threshold is None:
            ui_builtin_parsing_threshold = 5
        self.ui_builtin_parsing_threshold = ui_builtin_parsing_threshold
        self.intents_data_sizes = intents_data_sizes

    @property
    def builtin_parser(self):
        return self._builtin_parser

    @builtin_parser.setter
    def builtin_parser(self, value):
        if value is not None \
                and value.parser.language != self.language.iso_code:
            raise ValueError(
                "Built in parser language code ('%s') is different from "
                "provided language code ('%s')"
                % (value.parser.language, self.language.iso_code))
        self._builtin_parser = value

    def parse(self, text, intent=None, force_builtin_entities=False):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        If the intent is provided, intent classification is not performed.
        If the builtin entity parsing is enforced, then the intent must be 
        provided
        """
        if force_builtin_entities:
            if intent is None:
                raise ValueError("If builtin entities parsing if enforced, "
                                 "intent should be passed")
            return self._parse_and_force_builtin_entities(
                text, intent).as_dict()
        else:
            return self._parse(text, intent=intent).as_dict()

    def _parse(self, text, intent=None):
        result = _parse(text, self.entities, self.rule_based_parser,
                        self.probabilistic_parser, self.builtin_parser,
                        intent)
        if result.is_empty():
            return result

        result = self.augment_slots_with_builtin_entities(result)
        return result

    def _parse_and_force_builtin_entities(self, text, intent):
        """
        Parse the input text for UI auto tagging and returns a dictionary  
        containing the most likely slots.
        """
        result = self._parse(text, intent=intent)
        force_builtin_parsing = self.intents_data_sizes[intent] < \
                                self.ui_builtin_parsing_threshold
        if not force_builtin_parsing:
            return result

        built_in_entities = get_built_in_entities(text, self.language)
        if len(built_in_entities) == 0:
            return result

        slots = result.parsed_slots
        if slots is None:
            slots = [ParsedSlot(e[MATCH_RANGE], e[VALUE], e[ENTITY].label,
                                e[ENTITY].label) for e in built_in_entities]
        else:
            for ent in built_in_entities:
                if any(s.match_range[0] <= ent[MATCH_RANGE][1]
                       and s.match_range[1] >= ent[MATCH_RANGE][0]
                       for s in slots):
                    continue
                parsed_slot = ParsedSlot(ent[MATCH_RANGE], ent[VALUE],
                                         ent[ENTITY].label, ent[ENTITY].label)
                slots.append(parsed_slot)
        parsed_intent = IntentClassificationResult(
            result.parsed_intent.intent_name, result.parsed_intent.probability)
        return Result(text, parsed_intent=parsed_intent, parsed_slots=slots)

    def augment_slots_with_builtin_entities(self, result):
        if self.probabilistic_parser is None:
            return result

        intent_name = result.parsed_intent.intent_name
        intent_slots_mapping = self.slot_name_mapping.get(intent_name, dict())
        all_intent_slots = intent_slots_mapping.keys()
        builtin_slots = set(s for s in all_intent_slots
                            if intent_slots_mapping[s] in
                            BuiltInEntity.built_in_entity_by_label)
        found_slots = set(s.slot_name for s in result.parsed_slots)
        missing_builtin_slots = set(builtin_slots).difference(found_slots)
        if len(missing_builtin_slots) == 0:
            return result

        tagger = self.probabilistic_parser.crf_taggers[intent_name]
        text = result.text
        scope = [BuiltInEntity.from_label(intent_slots_mapping[slot])
                 for slot in missing_builtin_slots]
        builtin_entities = get_built_in_entities(text, self.language, scope)
        slots = augment_slots(text, tagger, intent_slots_mapping,
                              builtin_entities, missing_builtin_slots)
        return Result(text, parsed_intent=result.parsed_intent,
                      parsed_slots=slots)

    def fit(self, dataset):
        """
        Fit the engine with a dataset and return it
        :param dataset: A dictionary containing data of the custom and builtin 
        intents.
        See https://github.com/snipsco/snips-nlu/blob/develop/README.md for
        details about the format.
        :return: A fitted SnipsNLUEngine
        """
        dataset = validate_and_format_dataset(dataset)
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE)
        self.rule_based_parser = RegexIntentParser().fit(dataset)
        self.entities = snips_nlu_entities(dataset)
        self.intents_data_sizes = {intent_name: len(intent[UTTERANCES])
                                   for intent_name, intent
                                   in custom_dataset[INTENTS].iteritems()}
        self.slot_name_mapping = get_slot_name_mapping(custom_dataset)
        taggers = dict()
        for intent in custom_dataset[INTENTS]:
            intent_custom_entities = get_intent_custom_entities(custom_dataset,
                                                                intent)
            features = crf_features(intent_custom_entities, self.language)
            taggers[intent] = CRFTagger(default_crf_model(), features,
                                        TaggingScheme.BIO, self.language)
        intent_classifier = SnipsIntentClassifier(self.language)
        self.probabilistic_parser = ProbabilisticIntentParser(
            self.language, intent_classifier, taggers, self.slot_name_mapping)
        self.probabilistic_parser.fit(dataset)
        return self

    def to_dict(self):
        """
        Serialize the SnipsNLUEngine to a json dict, after having reset the
        builtin intent parser. Thus this serialization, contains only the
        custom intent parsers.
        """
        language_code = None
        if self.language is not None:
            language_code = self.language.iso_code

        rule_based_parser_dict = None
        probabilistic_parser_dict = None
        if self.rule_based_parser is not None:
            rule_based_parser_dict = self.rule_based_parser.to_dict()
        if self.probabilistic_parser is not None:
            probabilistic_parser_dict = self.probabilistic_parser.to_dict()

        return {
            LANGUAGE: language_code,
            "rule_based_parser": rule_based_parser_dict,
            "probabilistic_parser": probabilistic_parser_dict,
            BUILTIN_PARSER: None,
            "slot_name_mapping": self.slot_name_mapping,
            "ui_builtin_parsing_threshold": self.ui_builtin_parsing_threshold,
            ENTITIES: self.entities,
            "intents_data_sizes": self.intents_data_sizes
        }

    @classmethod
    def load_from(cls, language, customs=None, builtin_path=None,
                  builtin_binary=None):
        """
        Create a `SnipsNLUEngine` from the following attributes
        
        :param language: ISO 639-1 language code or Language instance
        :param customs: A `dict` containing custom intents data
        :param builtin_path: A directory path containing builtin intents data
        :param builtin_binary: A `bytearray` containing builtin intents data
        """

        if isinstance(language, (str, unicode)):
            language = Language.from_iso_code(language)

        rule_based_parser = None
        probabilistic_parser = None
        builtin_parser = None
        entities = None
        ui_builtin_parsing_threshold = None
        slot_name_mapping = None
        intent_data_size = None

        if customs is not None:
            rule_based_parser = instance_from_dict(
                customs["rule_based_parser"])
            probabilistic_parser = instance_from_dict(
                customs["probabilistic_parser"])
            entities = customs[ENTITIES]
            ui_builtin_parsing_threshold = customs[
                "ui_builtin_parsing_threshold"]
            slot_name_mapping = customs["slot_name_mapping"]
            intent_data_size = customs["intents_data_sizes"]

        if builtin_path is not None or builtin_binary is not None:
            builtin_parser = BuiltinIntentParser(language=language,
                                                 data_path=builtin_path,
                                                 data_binary=builtin_binary)

        return cls(language, rule_based_parser=rule_based_parser,
                   probabilistic_parser=probabilistic_parser,
                   builtin_parser=builtin_parser,
                   slot_name_mapping=slot_name_mapping,
                   entities=entities,
                   ui_builtin_parsing_threshold=ui_builtin_parsing_threshold,
                   intents_data_sizes=intent_data_size)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.to_dict() == other.to_dict()

    def __ne__(self, other):
        return not self.__eq__(other)
