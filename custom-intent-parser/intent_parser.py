from abc import ABCMeta, abstractmethod


class IntentParser(object):
    __metaclass__ = ABCMeta

    _intents = []
    _slots = []

    @property
    def intents(self):
        return self._intents

    @intents.setter
    def intents(self, value):
        self._intents = value

    @property
    def slots(self):
        return self._slots

    @slots.setter
    def slots(self, value):
        self._slots = value

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def get_slots(self, text, intent=None):
        pass

    @abstractmethod
    def parse(self, text):
        pass
