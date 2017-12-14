from snips_nlu.configs.config import Serializable
from snips_nlu.configs.slot_filler import CRFSlotFillerConfig
from snips_nlu.configs.intent_classifier import IntentClassifierConfig


class ProbabilisticIntentParserConfig(Serializable):
    def __init__(self, intent_classifier_config=None,
                 crf_slot_filler_config=None):
        if intent_classifier_config is None:
            intent_classifier_config = IntentClassifierConfig()
        if crf_slot_filler_config is None:
            crf_slot_filler_config = CRFSlotFillerConfig()
        self._intent_classifier_config = None
        self.intent_classifier_config = intent_classifier_config
        self._crf_slot_filler_config = None
        self.crf_slot_filler_config = crf_slot_filler_config

    @property
    def crf_slot_filler_config(self):
        return self._crf_slot_filler_config

    @crf_slot_filler_config.setter
    def crf_slot_filler_config(self, value):
        if isinstance(value, dict):
            self._crf_slot_filler_config = \
                CRFSlotFillerConfig.from_dict(value)
        elif isinstance(value, CRFSlotFillerConfig):
            self._crf_slot_filler_config = value
        else:
            raise TypeError("Expected instance of CRFSlotFillerConfig or dict "
                            "but received: %s" % type(value))

    @property
    def intent_classifier_config(self):
        return self._intent_classifier_config

    @intent_classifier_config.setter
    def intent_classifier_config(self, value):
        if isinstance(value, dict):
            self._intent_classifier_config = \
                IntentClassifierConfig.from_dict(value)
        elif isinstance(value, IntentClassifierConfig):
            self._intent_classifier_config = value
        else:
            raise TypeError("Expected instance of IntentClassifierConfig or "
                            "dict but received: %s" % type(value))

    def to_dict(self):
        return {
            "crf_slot_filler_config": self.crf_slot_filler_config.to_dict(),
            "intent_classifier_config": self.intent_classifier_config.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class DeterministicIntentParserConfig(Serializable):
    def __init__(self, max_queries=50, max_entities=200):
        self.max_queries = max_queries
        self.max_entities = max_entities

    def to_dict(self):
        return {
            "max_queries": self.max_queries,
            "max_entities": self.max_entities
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)
