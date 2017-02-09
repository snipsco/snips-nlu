from abc import ABCMeta, abstractmethod


class EntityExtractor(object):
    __metaclass__ = ABCMeta

    _entities = []

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, value):
        self._entities = value

    @abstractmethod
    def fit(self, queries):
        pass

    @abstractmethod
    def get_entities(self, text):
        pass
