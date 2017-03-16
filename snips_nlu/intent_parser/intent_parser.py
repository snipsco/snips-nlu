from abc import ABCMeta, abstractmethod

from ..result import intent_classification_result

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
        intent = parsing["intent"]
        prob = None
        if intent is not None:
            prob = intent["prob"]
            intent = intent["name"]
        return intent_classification_result(intent, prob)

    def get_entities(self, text, intent=None):
        parsing = self.parse(text)
        return parsing["entities"]
