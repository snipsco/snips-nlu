from __future__ import unicode_literals

from copy import deepcopy

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import ENTITIES, CAPITALIZE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.probabilistic_intent_parser import (
    ProbabilisticIntentParser)
from snips_nlu.nlu_engine.utils import parse
from snips_nlu.pipeline.configs.nlu_engine import NLUEngineConfig
from snips_nlu.pipeline.processing_unit import (
    ProcessingUnit, build_processing_unit, load_processing_unit)
from snips_nlu.version import __model_version__, __version__


class SnipsNLUEngine(ProcessingUnit):
    unit_name = "nlu_engine"
    config_type = NLUEngineConfig

    def __init__(self, config=None):
        if config is None:
            config = self.config_type()
        super(SnipsNLUEngine, self).__init__(config)
        self.intent_parsers = []
        self.entities = None

    def parse(self, text, intent=None):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        result = parse(text, self.entities, self.intent_parsers, intent)
        return result.as_dict()

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

        parsers = []
        for parser_config in self.config.intent_parsers_configs:
            # Re-use existing parsers to allow pre-training
            recycled_parser = None
            for parser in self.intent_parsers:
                if parser.unit_name == parser_config.unit_name:
                    recycled_parser = parser
                    break
            if recycled_parser is None:
                recycled_parser = build_processing_unit(parser_config)
            parsers.append(recycled_parser)

        self.intent_parsers = parsers

        for parser in self.intent_parsers:
            parser.fit(dataset, intents=intents)
        return self

    def get_fitted_slot_filler(self, dataset, intent):
        probabilistic_parser = None
        for intent_parser in self.intent_parsers:
            if intent_parser.unit_name == ProbabilisticIntentParser.unit_name:
                probabilistic_parser = intent_parser

        if probabilistic_parser is None:
            probabilistic_parser_config = None
            for parser_config in self.config.intent_parsers_configs:
                if parser_config.unit_name == \
                        ProbabilisticIntentParser.unit_name:
                    probabilistic_parser_config = parser_config
                    break
            probabilistic_parser = ProbabilisticIntentParser(
                probabilistic_parser_config)

        dataset = validate_and_format_dataset(dataset)
        return probabilistic_parser.get_fitted_slot_filler(dataset, intent)

    def add_fitted_slot_filler(self, intent, model_data):
        probabilistic_parser = None
        for intent_parser in self.intent_parsers:
            if intent_parser.unit_name == ProbabilisticIntentParser.unit_name:
                probabilistic_parser = intent_parser

        if probabilistic_parser is None:
            probabilistic_parser_config = None
            for parser_config in self.config.intent_parsers_configs:
                if parser_config.unit_name == \
                        ProbabilisticIntentParser.unit_name:
                    probabilistic_parser_config = parser_config
                    break
            probabilistic_parser = ProbabilisticIntentParser(
                probabilistic_parser_config)
            self.intent_parsers.append(probabilistic_parser)

        probabilistic_parser.add_fitted_slot_filler(intent, model_data)

    def to_dict(self):
        """
        Serialize the nlu engine into a python dictionary
        """
        intent_parsers = [parser.to_dict() for parser in self.intent_parsers]
        return {
            "unit_name": self.unit_name,
            ENTITIES: self.entities,
            "intent_parsers": intent_parsers,
            "config": self.config.to_dict(),
            "model_version": __model_version__,
            "training_package_version": __version__
        }

    @classmethod
    def from_dict(cls, unit_dict):
        """
        Loads a SnipsNLUEngine instance from a python dictionary.
        """
        model_version = unit_dict.get("model_version")
        if model_version is None or model_version != __model_version__:
            raise ValueError(
                "Incompatible data model: persisted object=%s, python lib=%s"
                % (model_version, __model_version__))

        nlu_engine = SnipsNLUEngine(config=unit_dict["config"])
        nlu_engine.entities = unit_dict[ENTITIES]
        nlu_engine.intent_parsers = [
            load_processing_unit(parser_dict)
            for parser_dict in unit_dict["intent_parsers"]]

        return nlu_engine
