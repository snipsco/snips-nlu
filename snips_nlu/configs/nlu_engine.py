from snips_nlu.configs.serializable import Serializable
from snips_nlu.configs.intent_parser import ProbabilisticIntentParserConfig, \
    DeterministicIntentParserConfig


class NLUEngineConfig(Serializable):
    def __init__(self, probabilistic_intent_parser_config=None,
                 deterministic_intent_parser_config=None):
        if probabilistic_intent_parser_config is None:
            probabilistic_intent_parser_config = \
                ProbabilisticIntentParserConfig()
        if deterministic_intent_parser_config is None:
            deterministic_intent_parser_config = \
                DeterministicIntentParserConfig()
        self._probabilistic_intent_parser_config = None
        self.probabilistic_intent_parser_config = \
            probabilistic_intent_parser_config
        self._deterministic_intent_parser_config = None
        self.deterministic_intent_parser_config = \
            deterministic_intent_parser_config

    @property
    def probabilistic_intent_parser_config(self):
        return self._probabilistic_intent_parser_config

    @probabilistic_intent_parser_config.setter
    def probabilistic_intent_parser_config(self, value):
        if isinstance(value, dict):
            self._probabilistic_intent_parser_config = \
                ProbabilisticIntentParserConfig.from_dict(value)
        elif isinstance(value, ProbabilisticIntentParserConfig):
            self._probabilistic_intent_parser_config = value
        else:
            raise TypeError("Expected instance of "
                            "ProbabilisticIntentParserConfig or dict but "
                            "received: %s" % type(value))

    @property
    def deterministic_intent_parser_config(self):
        return self._deterministic_intent_parser_config

    @deterministic_intent_parser_config.setter
    def deterministic_intent_parser_config(self, value):
        if isinstance(value, dict):
            self._deterministic_intent_parser_config = \
                DeterministicIntentParserConfig.from_dict(value)
        elif isinstance(value, DeterministicIntentParserConfig):
            self._deterministic_intent_parser_config = value
        else:
            raise TypeError("Expected instance of "
                            "DeterministicIntentParserConfig or dict but "
                            "received: %s" % type(value))

    def to_dict(self):
        return {
            "probabilistic_intent_parser_config":
                self.probabilistic_intent_parser_config.to_dict(),
            "deterministic_intent_parser_config":
                self.deterministic_intent_parser_config.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)
