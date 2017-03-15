import os
from abc import ABCMeta, abstractmethod

from ..intent_parser.builtin_intent_parser import BuiltinIntentParser
from ..intent_parser.intent_parser import IntentParser
from ..intent_parser.regex_intent_parser import RegexIntentParser
from ..result import result


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        pass


class SnipsNLUEngine(NLUEngine):
    def __init__(self, custom_intent_parser, builtin_intent_parser,
                 custom_first=True):
        super(SnipsNLUEngine, self).__init__()
        self._custom_intent_parser = None
        self.custom_intent_parser = custom_intent_parser
        self._builtin_intent_parser = None
        self.builtin_intent_parser = builtin_intent_parser
        self._built_in_intents = []
        self._fitted = False
        self.custom_first = custom_first

    @property
    def custom_intent_parser(self):
        return self._custom_intent_parser

    @custom_intent_parser.setter
    def custom_intent_parser(self, value):
        if value is not None and not isinstance(value, IntentParser):
            raise ValueError("Expected IntentParser, found: %s" % type(value))
        self._custom_intent_parser = value

    @property
    def builtin_intent_parser(self):
        return self._builtin_intent_parser

    @builtin_intent_parser.setter
    def builtin_intent_parser(self, value):
        if value is not None and not isinstance(value, IntentParser):
            raise ValueError("Expected IntentParser, found: %s" % type(value))
        self._builtin_intent_parser = value

    def parse(self, text):
        if self.custom_first:
            first_parser = self.custom_intent_parser
            second_parser = self.builtin_intent_parser
        else:
            first_parser = self.builtin_intent_parser
            second_parser = self.custom_intent_parser

        first_parse = self._parse(text, first_parser)
        if first_parse["intent"] is not None:
            return first_parse
        else:
            return self._parse(text, second_parser)

    @staticmethod
    def _parse(text, parser):
        if parser is not None:
            return parser.parse(text)
        else:
            return result(text)

    @classmethod
    def load(cls, directory_path):
        custom_intents_directory = os.path.join(directory_path,
                                                "custom_intents")
        builtin_intents_directory = os.path.join(directory_path,
                                                 "builtin_intents")
        if len(os.listdir(custom_intents_directory)) > 0:
            custom_intent_parser = RegexIntentParser.load(
                custom_intents_directory)
        else:
            custom_intent_parser = None
        if len(os.listdir(builtin_intents_directory)) > 0:
            builtin_intent_parser = BuiltinIntentParser(
                builtin_intents_directory)
        else:
            builtin_intent_parser = None
        return SnipsNLUEngine(custom_intent_parser, builtin_intent_parser)
