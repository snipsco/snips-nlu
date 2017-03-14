from abc import ABCMeta, abstractmethod

from intent_parser import IntentParser
from snips_nlu.utils import abstractclassmethod


class CustomIntentParser(IntentParser):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def fit(self, dataset):
        pass

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value):
        self._fitted = value

    @abstractmethod
    def save(self, path):
        pass

    @abstractclassmethod
    def load(cls, path):
        pass
