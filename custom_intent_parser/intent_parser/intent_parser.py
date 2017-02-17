from abc import ABCMeta, abstractmethod, abstractproperty

from data_helpers import is_existing_built_in
from utils import abstractclassmethod


class IntentParser(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def fitted(self):
        pass

    @fitted.setter
    def fitted(self, value):
        pass

    def check_fitted(self):
        if not self.fitted:
            raise ValueError("IntentParser must be fitted before calling the"
                             " 'fit' method.")

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def parse(self, text):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def get_entities(self, text, intent=None):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractclassmethod
    def load(cls, path):
        pass

    @abstractclassmethod
    def from_dataset(cls, dataset):
        pass


class SnipsIntentParser(IntentParser):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(SnipsIntentParser, self).__init__()
        self._built_in_intents = []
        self._fitted = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, newvalue):
        self._value = newvalue

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value):
        self._fitted = value

    @property
    def built_in_intents(self):
        return self._built_in_intents

    @built_in_intents.setter
    def built_in_intents(self, value):
        for intent_name in value:
            if not is_existing_built_in(intent_name):
                raise AttributeError("Unknown built in intent: %s"
                                     % intent_name)
        self._built_in_intents = value
        self.fitted = False
