from abc import ABCMeta, abstractmethod

from dataset import validate_and_format_dataset, filter_dataset
from snips_nlu.built_in_entities import BuiltInEntity, get_built_in_entities
from snips_nlu.constants import (
    USE_SYNONYMS, SYNONYMS, DATA, INTENTS, ENTITIES, UTTERANCES,
    LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE, ENTITY, BUILTIN_PARSER,
    CUSTOM_PARSERS, CUSTOM_ENGINE, MATCH_RANGE)
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.builtin_intent_parser import BuiltinIntentParser
from snips_nlu.intent_parser.crf_intent_parser import CRFIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
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
        return Result(text, parsed_intent=None, parsed_slots=None)
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
    return Result(text, parsed_intent=None, parsed_slots=None)


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
    def __init__(self, language, builtin_parser=None, custom_parsers=None,
                 entities=None):
        super(SnipsNLUEngine, self).__init__(language)
        self._builtin_parser = None
        self.builtin_parser = builtin_parser
        self.custom_parsers = custom_parsers
        self.entities = entities

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

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        if self.builtin_parser is None and self.custom_parsers is None:
            raise ValueError("NLUEngine as no built-in parser nor "
                             "custom parsers")
        parsers = []
        if self.custom_parsers is not None:
            parsers += self.custom_parsers
        if self.builtin_parser is not None:
            parsers.append(self.builtin_parser)

        return _parse(text, parsers, self.entities).as_dict()

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
        custom_parser = RegexIntentParser().fit(dataset)
        self.entities = snips_nlu_entities(dataset)
        taggers = dict()
        for intent in custom_dataset[INTENTS].keys():
            intent_custom_entities = get_intent_custom_entities(custom_dataset,
                                                                intent)
            features = crf_features(intent_custom_entities,
                                    language=self.language)
            taggers[intent] = CRFTagger(default_crf_model(), features,
                                        TaggingScheme.BIO, self.language)
        intent_classifier = SnipsIntentClassifier(self.language)
        crf_parser = CRFIntentParser(self.language, intent_classifier, taggers)
        crf_parser = crf_parser.fit(dataset)
        self.custom_parsers = [custom_parser, crf_parser]
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

        return {
            LANGUAGE: language_code,
            CUSTOM_PARSERS: [p.to_dict() for p in self.custom_parsers],
            BUILTIN_PARSER: None,
            ENTITIES: self.entities
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

        custom_parsers = None
        entities = None
        if customs is not None:
            custom_parsers = [instance_from_dict(d) for d in
                              customs[CUSTOM_PARSERS]]
            entities = customs[ENTITIES]
        builtin_parser = None
        if builtin_path is not None or builtin_binary is not None:
            builtin_parser = BuiltinIntentParser(language=language,
                                                 data_path=builtin_path,
                                                 data_binary=builtin_binary)

        return cls(language, builtin_parser=builtin_parser,
                   custom_parsers=custom_parsers, entities=entities)

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
