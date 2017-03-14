from abc import ABCMeta, abstractmethod

from ..intent_parser.builtin_intent_parser import BuiltinIntentParser
from ..intent_parser.custom_intent_parser import CustomIntentParser
from ..result import result


class IntentOrchestrator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def orchestrate_parsing(self, text, custom_intent_parser,
                            builtin_intent_parser):
        pass


class SimplePriorityOrchestrator(IntentOrchestrator):
    def __init__(self, custom_first):
        """
        This orchestrator will return the most likely result from :
            - the custom intent parser if custom_first is True
            - the builtin intent parser if custom_first is False
        """
        self.custom_first = custom_first

    def orchestrate_parsing(self, text, custom_intent_parser,
                            builtin_intent_parser):
        if custom_intent_parser is not None and not isinstance(
                custom_intent_parser, CustomIntentParser):
            raise ValueError("Expected CustomIntentParser, found: %s"
                             % type(custom_intent_parser))

        if builtin_intent_parser is not None and not isinstance(
                builtin_intent_parser, BuiltinIntentParser):
            raise ValueError("Expected BuiltinIntentParser, found: %s"
                             % type(custom_intent_parser))

        if self.custom_first:
            first_parser = custom_intent_parser
            second_parser = builtin_intent_parser
        else:
            first_parser = builtin_intent_parser
            second_parser = custom_intent_parser

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
