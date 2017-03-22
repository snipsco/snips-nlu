from abc import ABCMeta, abstractmethod


class CustomIntentParser(object):
    __metaclass__ = ABCMeta

    def __init__(self, intent_name):
        self.intent_name = intent_name

    @abstractmethod
    def parse(self, text):
        raise NotImplementedError

    @abstractmethod
    def get_intent(self, text):
        raise NotImplementedError

    @abstractmethod
    def get_entities(self, text, intent=None):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)
