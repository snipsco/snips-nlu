from abc import ABCMeta, abstractmethod, abstractproperty

from snips_nlu.utils import abstractclassmethod


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def fitted(self):
        pass

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractclassmethod
    def from_dict(cls, obj_dict):
        raise NotImplementedError
