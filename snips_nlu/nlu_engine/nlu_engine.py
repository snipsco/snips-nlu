from __future__ import unicode_literals

from copy import deepcopy

from future.utils import iteritems

from snips_nlu.builtin_entities import is_builtin_entity, BuiltInEntity
from snips_nlu.constants import (
    ENTITIES, CAPITALIZE, LANGUAGE, RES_SLOTS, RES_ENTITY, RES_INTENT)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.nlu_engine.utils import resolve_slots
from snips_nlu.pipeline.configs.nlu_engine import NLUEngineConfig
from snips_nlu.pipeline.processing_unit import (
    ProcessingUnit, build_processing_unit, load_processing_unit)
from snips_nlu.result import empty_result, is_empty, parsing_result
from snips_nlu.utils import get_slot_name_mappings, NotTrained
from snips_nlu.version import __model_version__, __version__


class SnipsNLUEngine(ProcessingUnit):
    unit_name = "nlu_engine"
    config_type = NLUEngineConfig

    def __init__(self, config=None):
        if config is None:
            config = self.config_type()
        super(SnipsNLUEngine, self).__init__(config)
        self.intent_parsers = []
        self._dataset_metadata = None

    @property
    def fitted(self):
        return self._dataset_metadata is not None

    def fit(self, dataset, force_retrain=True):
        """Fit the NLU engine

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
        self._dataset_metadata = get_dataset_metadata(dataset)

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
            if force_retrain or not recycled_parser.fitted:
                recycled_parser.fit(dataset)
            parsers.append(recycled_parser)

        self.intent_parsers = parsers
        return self

    def parse(self, text, intents=None):
        """Parse the input text and returns a dictionary containing the most
            likely intent and slots
        """
        if not self.fitted:
            raise NotTrained("SnipsNLUEngine must be fitted")

        if isinstance(intents, str):
            intents = [intents]

        language = Language.from_iso_code(
            self._dataset_metadata["language_code"])
        entities = self._dataset_metadata["entities"]

        for parser in self.intent_parsers:
            res = parser.parse(text, intents)
            if is_empty(res):
                continue
            slots = res[RES_SLOTS]
            scope = [BuiltInEntity.from_label(s[RES_ENTITY]) for s in slots
                     if is_builtin_entity(s[RES_ENTITY])]
            resolved_slots = resolve_slots(text, slots, entities, language,
                                           scope)
            return parsing_result(text, intent=res[RES_INTENT],
                                  slots=resolved_slots)
        return empty_result(text)

    def to_dict(self):
        """
        Serialize the nlu engine into a python dictionary
        """
        intent_parsers = [parser.to_dict() for parser in self.intent_parsers]
        return {
            "unit_name": self.unit_name,
            "dataset_metadata": self._dataset_metadata,
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

        nlu_engine = cls(config=unit_dict["config"])
        nlu_engine._dataset_metadata = unit_dict["dataset_metadata"]
        nlu_engine.intent_parsers = [
            load_processing_unit(parser_dict)
            for parser_dict in unit_dict["intent_parsers"]]

        return nlu_engine


def get_dataset_metadata(dataset):
    entities = dict()
    for entity_name, entity in iteritems(dataset[ENTITIES]):
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
