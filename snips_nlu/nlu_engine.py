from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from copy import deepcopy

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.config import NLUConfig
from snips_nlu.constants import (
    INTENTS, ENTITIES, UTTERANCES, AUTOMATICALLY_EXTENSIBLE,
    ENTITY, DATA, SLOT_NAME, CAPITALIZE)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot, empty_result, \
    IntentClassificationResult
from snips_nlu.result import Result
from snips_nlu.version import __model_version__, __version__


class NLUEngine(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._language = None

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
    def parse(self, text, intent=None):
        pass


def _parse(text, entities, rule_based_parser=None, probabilistic_parser=None,
           intent=None):
    parsers = []
    if rule_based_parser is not None:
        parsers.append(rule_based_parser)
    if probabilistic_parser is not None:
        parsers.append(probabilistic_parser)

    if not parsers:
        return empty_result(text)

    result = empty_result(text) if intent is None else Result(
        text, parsed_intent=IntentClassificationResult(intent, 1.0),
        parsed_slots=[])

    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue

        intent_name = res.intent_name
        if intent is not None:
            if intent_name != intent:
                continue
            res = IntentClassificationResult(intent_name, 1.0)

        valid_slot = []
        slots = parser.get_slots(text, intent_name)
        for s in slots:
            slot_value = s.value
            # Check if the entity is from a custom intent
            if s.entity in entities:
                entity = entities[s.entity]
                if s.value in entity[UTTERANCES]:
                    slot_value = entity[UTTERANCES][s.value]
                elif not entity[AUTOMATICALLY_EXTENSIBLE]:
                    continue
            s = ParsedSlot(s.match_range, slot_value, s.entity,
                           s.slot_name)
            valid_slot.append(s)
        return Result(text, parsed_intent=res, parsed_slots=valid_slot)
    return result


def get_intent_slot_name_mapping(dataset, intent):
    slot_name_mapping = dict()
    intent_data = dataset[INTENTS][intent]
    for utterance in intent_data[UTTERANCES]:
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name_mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def enrich_slots(slots, other_slots):
    enriched_slots = list(slots)
    for slot in other_slots:
        if any((slot.match_range[1] > s.match_range[0])
               and (slot.match_range[0] < s.match_range[1])
               for s in enriched_slots):
            continue
        enriched_slots.append(slot)
    return enriched_slots


class SnipsNLUEngine(NLUEngine):
    def __init__(self, config=NLUConfig()):
        super(SnipsNLUEngine, self).__init__()
        self._config = None
        self.config = config
        self.rule_based_parser = None
        self.probabilistic_parser = None
        self.entities = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, NLUConfig):
            config = value
        elif isinstance(value, dict):
            config = NLUConfig.from_dict(value)
        else:
            raise TypeError("Expected config to be a dict or a NLUConfig")
        self._config = config

    def parse(self, text, intent=None):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        return self._parse(text, intent=intent).as_dict()

    def _parse(self, text, intent=None):
        return _parse(text, self.entities, self.rule_based_parser,
                      self.probabilistic_parser, intent)

    def fit(self, dataset, intents=None):

        """
        Fit the NLU engine.

        Parameters
        ----------
        - dataset: dict containing intents and entities data
        - intents: list of intents to train. If `None`, all intents will be
        trained. This parameter allows to have pre-trained intents.

        Returns
        -------
        The same object, trained
        """
        dataset = validate_and_format_dataset(dataset)
        self.entities = dict()
        for entity_name, entity in dataset[ENTITIES].iteritems():
            if is_builtin_entity(entity_name):
                continue
            ent = deepcopy(entity)
            ent.pop(CAPITALIZE)
            self.entities[entity_name] = ent

        self.rule_based_parser = RegexIntentParser().fit(dataset)

        if self.probabilistic_parser is None:
            self.probabilistic_parser = ProbabilisticIntentParser(
                self.config.probabilistic_intent_parser_config)
        self.probabilistic_parser.fit(dataset, intents=intents)
        return self

    def get_fitted_tagger(self, dataset, intent):
        dataset = validate_and_format_dataset(dataset)
        if self.probabilistic_parser is None:
            self.probabilistic_parser = ProbabilisticIntentParser(
                self.config.probabilistic_intent_parser_config)
        return self.probabilistic_parser.get_fitted_slot_filler(dataset,
                                                                intent)

    def add_fitted_tagger(self, intent, model_data):
        if self.probabilistic_parser is None:
            self.probabilistic_parser = ProbabilisticIntentParser(
                self.config.probabilistic_intent_parser_config)
        self.probabilistic_parser.add_fitted_slot_filler(intent, model_data)

    def to_dict(self):
        """
        Serialize the nlu engine into a python dictionary
        """
        model_dict = dict()
        if self.rule_based_parser is not None:
            model_dict["rule_based_parser"] = self.rule_based_parser.to_dict()
        if self.probabilistic_parser is not None:
            model_dict["probabilistic_parser"] = \
                self.probabilistic_parser.to_dict()

        return {
            ENTITIES: self.entities,
            "model": model_dict,
            "config": self.config.to_dict(),
            "model_version": __model_version__,
            "training_package_version": __version__
        }

    @classmethod
    def from_dict(cls, obj_dict):
        """
        Loads a SnipsNLUEngine instance from a python dictionary.
        """
        model_version = obj_dict.get("model_version")
        if model_version is None or model_version != __model_version__:
            raise ValueError(
                "Incompatible data model: persisted object=%s, python lib=%s"
                % (model_version, __model_version__))

        nlu_engine = SnipsNLUEngine(config=obj_dict["config"])
        nlu_engine.entities = obj_dict[ENTITIES]

        if "rule_based_parser" in obj_dict["model"]:
            nlu_engine.rule_based_parser = RegexIntentParser.from_dict(
                obj_dict["model"]["rule_based_parser"])

        if "probabilistic_parser" in obj_dict["model"]:
            nlu_engine.probabilistic_parser = \
                ProbabilisticIntentParser.from_dict(
                    obj_dict["model"]["probabilistic_parser"])

        return nlu_engine
