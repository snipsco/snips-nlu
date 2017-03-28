from abc import ABCMeta, abstractmethod, abstractproperty

from snips_nlu.result import IntentClassificationResult
from snips_nlu.utils import abstractclassmethod, instance_from_dict, \
    instance_to_generic_dict


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
        return instance_from_dict(obj_dict)


class SnipsIntentClassifier(IntentClassifier):
    def __init__(self, intent_name=None):
        self.intent_name = intent_name

    @property
    def fitted(self):
        return self.intent_name is not None

    def fit(self, dataset):
        intents = dataset["intents"]
        if len(intents.keys()) > 0:
            self.intent_name = intents.keys()[0]
        return self

    def get_intent(self, text):
        if self.intent_name is not None:
            return IntentClassificationResult(intent_name=self.intent_name,
                                              probability=1.0)
        return None

    def to_dict(self):
        obj_dict = instance_to_generic_dict(self)
        obj_dict.update({"intent_name": self.intent_name})
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        return SnipsIntentClassifier(intent_name=obj_dict["intent_name"])
