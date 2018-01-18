from copy import deepcopy

from snips_nlu.pipeline.configs.config import ProcessingUnitConfig
from snips_nlu.pipeline.configs.intent_classifier import IntentClassifierConfig
from snips_nlu.pipeline.configs.slot_filler import CRFSlotFillerConfig
from snips_nlu.pipeline.processing_unit import get_processing_unit_config
from snips_nlu.utils import classproperty


class ProbabilisticIntentParserConfig(ProcessingUnitConfig):
    def __init__(self, intent_classifier_config=None, slot_filler_config=None):
        if intent_classifier_config is None:
            intent_classifier_config = IntentClassifierConfig()
        if slot_filler_config is None:
            slot_filler_config = CRFSlotFillerConfig()
        self.intent_classifier_config = get_processing_unit_config(
            intent_classifier_config)
        self.slot_filler_config = get_processing_unit_config(
            slot_filler_config)

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.intent_parser.probabilistic_intent_parser import \
            ProbabilisticIntentParser
        return ProbabilisticIntentParser.unit_name

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "slot_filler_config": self.slot_filler_config.to_dict(),
            "intent_classifier_config": self.intent_classifier_config.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)


class DeterministicIntentParserConfig(ProcessingUnitConfig):
    def __init__(self, max_queries=50, max_entities=200):
        self.max_queries = max_queries
        self.max_entities = max_entities

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.intent_parser.deterministic_intent_parser import \
            DeterministicIntentParser
        return DeterministicIntentParser.unit_name

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "max_queries": self.max_queries,
            "max_entities": self.max_entities
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)
