import os
from abc import ABCMeta, abstractmethod

from ..intent_parser.builtin_intent_parser import BuiltinIntentParser
from ..intent_parser.intent_parser import CUSTOM_PARSER_TYPE, \
    BUILTIN_PARSER_TYPE
from ..intent_parser.regex_intent_parser import RegexIntentParser
from ..result import result


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        pass


class SnipsNLUEngine(NLUEngine):
    def __init__(self, parsers, custom_first=True):
        super(SnipsNLUEngine, self).__init__()
        self.custom_parsers = filter(
            lambda parser: parser.parser_type == CUSTOM_PARSER_TYPE, parsers)
        self.builtin_parsers = filter(
            lambda parser: parser.parser_type == BUILTIN_PARSER_TYPE, parsers)
        self.custom_first = custom_first

    def parse(self, text):
        if self.custom_first:
            first_parsers = self.custom_parsers
            second_parsers = self.builtin_parsers
        else:
            first_parsers = self.builtin_parsers
            second_parsers = self.custom_parsers

        first_parse = self._parse(text, first_parsers)
        if first_parse["intent"] is not None:
            return first_parse
        else:
            return self._parse(text, second_parsers)

    @staticmethod
    def _parse(text, parsers):
        if len(parsers) == 0:
            return result(text)

        best_parser = None
        best_intent = None
        for parser in parsers:
            res = parser.get_intent(text)
            if best_intent is None or res["prob"] > best_intent["prob"]:
                best_intent = res
                best_parser = parser
        entities = best_parser.get_entities(text)
        return result(text, best_intent, entities)

    @classmethod
    def load(cls, path):
        custom_intents_dir = os.path.join(path, "custom_intents")
        builtin_intents_dir = os.path.join(path, "builtin_intents")
        custom_parsers = [
            RegexIntentParser.load(os.path.join(custom_intents_dir, path)) for
            path in os.listdir(custom_intents_dir)]
        configs_dir = os.path.join(builtin_intents_dir, 'configurations')
        gazetteers_dir = os.path.join(builtin_intents_dir, 'gazetteers')
        builtin_parsers = [BuiltinIntentParser(config_path, gazetteers_dir) for
                           config_path in os.listdir(configs_dir)]
        return SnipsNLUEngine(custom_parsers + builtin_parsers)
