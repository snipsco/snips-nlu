from __future__ import unicode_literals

from copy import deepcopy

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import ENTITIES, CAPITALIZE, LANGUAGE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.nlu_engine.utils import parse
from snips_nlu.pipeline.configs.nlu_engine import NLUEngineConfig
from snips_nlu.pipeline.processing_unit import (
    ProcessingUnit, build_processing_unit, load_processing_unit)
from snips_nlu.utils import get_slot_name_mappings
from snips_nlu.version import __model_version__, __version__


class SnipsNLUEngine(ProcessingUnit):
    unit_name = "nlu_engine"
    config_type = NLUEngineConfig

    def __init__(self, config=None):
        if config is None:
            config = self.config_type()
        super(SnipsNLUEngine, self).__init__(config)
        self.intent_parsers = []
        self.dataset_metadata = None

    @property
    def fitted(self):
        return self.dataset_metadata is not None

    def parse(self, text, intent=None):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        if not self.fitted:
            raise AssertionError("NLU engine must be fitted before calling "
                                 "`parse`")
        language = Language.from_iso_code(
            self.dataset_metadata["language_code"])
        return parse(text, self.dataset_metadata["entities"], language,
                     self.intent_parsers, intent)

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
        self.dataset_metadata = get_dataset_metadata(dataset)

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

    def to_dict(self):
        """
        Serialize the nlu engine into a python dictionary
        """
        intent_parsers = [parser.to_dict() for parser in self.intent_parsers]
        return {
            "unit_name": self.unit_name,
            "dataset_metadata": self.dataset_metadata,
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
        nlu_engine.dataset_metadata = unit_dict["dataset_metadata"]
        nlu_engine.intent_parsers = [
            load_processing_unit(parser_dict)
            for parser_dict in unit_dict["intent_parsers"]]

        return nlu_engine


def get_dataset_metadata(dataset):
    entities = dict()
    for entity_name, entity in dataset[ENTITIES].items():
        if is_builtin_entity(entity_name):
            continue
        ent = deepcopy(entity)
        ent.pop(CAPITALIZE)
        entities[entity_name] = ent
    slot_name_mappings = get_slot_name_mappings(dataset)
    return {
        "language_code": dataset[LANGUAGE],
        "entities": entities,
        "slot_name_mappings": slot_name_mappings
    }
