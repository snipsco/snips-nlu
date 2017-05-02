from abc import ABCMeta, abstractmethod

from dataset import validate_and_format_dataset, filter_dataset
from snips_nlu.built_in_entities import BuiltInEntity, get_built_in_entities
from snips_nlu.constants import (
    INTENTS, ENTITIES, UTTERANCES, LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE,
    ENTITY, BUILTIN_PARSER, CUSTOM_ENGINE, MATCH_RANGE, RULE_BASED_PARSER,
    PROBABILISTIC_PARSER, INTENTS_DATA_SIZES, SLOT_NAME_MAPPING,
    SMALL_DATA_REGIME_THRESHOLD)
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.builtin_intent_parser import BuiltinIntentParser
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
from snips_nlu.nlu_engine_utils import (augment_slots, get_slot_name_mapping,
                                        get_intent_custom_entities,
                                        snips_nlu_entities, empty_result)
from snips_nlu.result import ParsedSlot
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_functions import crf_features
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


def _parse(text, parsers, entities):
    if len(parsers) == 0:
        return empty_result(text)
    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue
        slots = parser.get_slots(text, res.intent_name)
        valid_slot = []
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


class SnipsNLUEngine(NLUEngine):
    def __init__(self, language, rule_based_parser=None,
                 probabilistic_parser=None, builtin_parser=None, entities=None,
                 intents_data_sizes=None, slot_name_mapping=None,
                 small_data_regime_threshold=20):
        super(SnipsNLUEngine, self).__init__(language)
        self.rule_based_parser = rule_based_parser
        self.probabilistic_parser = probabilistic_parser
        self._builtin_parser = None
        self.builtin_parser = builtin_parser
        self.entities = entities
        self._intents_data_sizes = None
        self.intents_data_sizes = intents_data_sizes
        self.slot_name_mapping = slot_name_mapping
        self.small_data_regime_threshold = small_data_regime_threshold

    @property
    def parsers(self):
        parsers = []
        if self.rule_based_parser is not None:
            parsers.append(self.rule_based_parser)
        if self.probabilistic_parser is not None:
            parsers.append(self.probabilistic_parser)
        if self.builtin_parser is not None:
            parsers.append(self.builtin_parser)
        return parsers

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

    @property
    def intents_data_sizes(self):
        return self._intents_data_sizes

    @intents_data_sizes.setter
    def intents_data_sizes(self, value):
        if value is None:
            self._intents_data_sizes = dict()
        elif isinstance(value, dict):
            self._intents_data_sizes = value
        else:
            raise TypeError("Expected dict but found %s: " % type(value))

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        result = _parse(text, self.parsers, self.entities)
        if result.is_empty():
            return result.as_dict()

        intent_name = result.parsed_intent.intent_name
        intent_nb_utterances = self.intents_data_sizes.get(intent_name, None)
        if intent_nb_utterances is not None \
                and intent_nb_utterances <= self.small_data_regime_threshold:
            result = self.augment_slots_with_builtin_entities(result)
        return result.as_dict()

    def augment_slots_with_builtin_entities(self, result):
        if self.probabilistic_parser is None:
            return result

        intent_name = result.parsed_intent.intent_name
        intent_slots_mapping = self.slot_name_mapping[intent_name]
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
            RULE_BASED_PARSER: rule_based_parser_dict,
            PROBABILISTIC_PARSER: probabilistic_parser_dict,
            BUILTIN_PARSER: None,
            ENTITIES: self.entities,
            INTENTS_DATA_SIZES: self.intents_data_sizes,
            SLOT_NAME_MAPPING: self.slot_name_mapping,
            SMALL_DATA_REGIME_THRESHOLD: self.small_data_regime_threshold
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
        intents_data_sizes = None
        slot_name_mapping = None
        small_data_regime_threshold = None

        if customs is not None:
            rule_based_parser = instance_from_dict(customs[RULE_BASED_PARSER])
            probabilistic_parser = instance_from_dict(
                customs[PROBABILISTIC_PARSER])
            entities = customs[ENTITIES]
            intents_data_sizes = customs[INTENTS_DATA_SIZES]
            slot_name_mapping = customs[SLOT_NAME_MAPPING]
            small_data_regime_threshold = customs[SMALL_DATA_REGIME_THRESHOLD]

        if builtin_path is not None or builtin_binary is not None:
            builtin_parser = BuiltinIntentParser(language=language,
                                                 data_path=builtin_path,
                                                 data_binary=builtin_binary)

        return cls(language, rule_based_parser=rule_based_parser,
                   probabilistic_parser=probabilistic_parser,
                   builtin_parser=builtin_parser, entities=entities,
                   intents_data_sizes=intents_data_sizes,
                   slot_name_mapping=slot_name_mapping,
                   small_data_regime_threshold=small_data_regime_threshold)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.to_dict() == other.to_dict()

    def __ne__(self, other):
        return not self.__eq__(other)


class BuiltInEntitiesNLUEngine(NLUEngine):
    def __init__(self, language):
        super(BuiltInEntitiesNLUEngine, self).__init__(language)

    def parse(self, text):
        built_in_entities = get_built_in_entities(text, self.language)
        slots = None
        if len(built_in_entities) > 0:
            slots = [
                ParsedSlot(match_range=e[MATCH_RANGE], value=e[VALUE],
                           entity=e[ENTITY].label, slot_name=e[ENTITY].label)
                for e in built_in_entities]
        return Result(text, parsed_intent=None, parsed_slots=slots).as_dict()
