from abc import ABCMeta, abstractmethod

from ..intent_orchestrator.intent_orchestrator import IntentOrchestrator
from ..intent_parser.builtin_intent_parser import BuiltinIntentParser
from ..intent_parser.custom_intent_parser import CustomIntentParser


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        pass


class SnipsNLUEngine(NLUEngine):
    def __init__(self, custom_intent_parser, builtin_intent_parser,
                 intent_orchestrator):
        super(SnipsNLUEngine, self).__init__()
        self._custom_intent_parser = None
        self.custom_intent_parser = custom_intent_parser

        self._builtin_intent_parser = None
        self.builtin_intent_parser = builtin_intent_parser

        self._intent_orchestrator = None
        self.intent_orchestrator = intent_orchestrator

        self._built_in_intents = []
        self._fitted = False

    @property
    def custom_intent_parser(self):
        return self._custom_intent_parser

    @custom_intent_parser.setter
    def custom_intent_parser(self, value):
        if value is not None and not isinstance(value, CustomIntentParser):
            raise ValueError("Expected CustomIntentParser, found: %s"
                             % type(value))
        self._custom_intent_parser = value

    @property
    def builtin_intent_parser(self):
        return self._builtin_intent_parser

    @builtin_intent_parser.setter
    def builtin_intent_parser(self, value):
        if value is not None and not isinstance(value, BuiltinIntentParser):
            raise ValueError("Expected BuiltinIntentParser, found: %s"
                             % type(value))
        self._builtin_intent_parser = value

    @property
    def intent_orchestrator(self):
        return self._intent_orchestrator

    @intent_orchestrator.setter
    def intent_orchestrator(self, value):
        if not isinstance(value, IntentOrchestrator):
            raise ValueError("Expected IntentOrchestrator, found: %s"
                             % type(value))
        self._intent_orchestrator = value

    def parse(self, text):
        return self.intent_orchestrator.orchestrate_parsing(
            text,
            self.custom_intent_parser,
            self.builtin_intent_parser
        )

    def save(self, path):
        pass

    def load(cls, path):
        pass
