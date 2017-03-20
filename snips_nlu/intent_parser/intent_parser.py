from abc import ABCMeta, abstractmethod

from ..result import IntentClassificationResult

CUSTOM_PARSER_TYPE, BUILTIN_PARSER_TYPE = ("CUSTOM", "BUILTIN")


class IntentParser(object):
    __metaclass__ = ABCMeta

    def __init__(self, intent_name, parser_type):
        self.intent_name = intent_name
        self.parser_type = parser_type

    @abstractmethod
    def parse(self, text):
        pass

    def get_intent(self, text):
        parsing = self.parse(text)
        intent = parsing.parsed_intent
        proba = None
        if intent is not None:
            proba = intent.probability
            intent = intent.name
        return IntentClassificationResult(intent_name=intent, probability=proba)

    def get_entities(self, text, intent=None):
        parsing = self.parse(text)
        return parsing.parsed_entities

