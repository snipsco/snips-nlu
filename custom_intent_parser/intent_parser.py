from abc import ABCMeta, abstractmethod


class IntentParser(object):
    __metaclass__ = ABCMeta

    _intents = []
    _entities = []

    @property
    def intents(self):
        return self._intents

    @intents.setter
    def intents(self, value):
        self._intents = value

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, value):
        self._entities = value

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def get_entities(self, text, intent=None):
        pass

    @abstractmethod
    def parse(self, text):
        pass
