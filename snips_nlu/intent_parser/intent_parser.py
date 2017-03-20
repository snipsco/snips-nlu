from abc import ABCMeta, abstractmethod

CUSTOM_PARSER_TYPE, BUILTIN_PARSER_TYPE = ("CUSTOM", "BUILTIN")


class IntentParser(object):
    __metaclass__ = ABCMeta

    def __init__(self, intent_name, parser_type):
        self.intent_name = intent_name
        self.parser_type = parser_type

    @abstractmethod
    def parse(self, text):
        raise NotImplementedError

    @abstractmethod
    def get_intent(self, text):
        raise NotImplementedError

    @abstractmethod
    def get_entities(self, text, intent=None):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)
