from abc import ABCMeta, abstractmethod, abstractproperty

from snips_nlu.utils import instance_from_dict


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

    @classmethod
    def from_dict(cls, obj_dict):
        return instance_from_dict(obj_dict)
