from abc import ABCMeta, abstractmethod

from snips_nlu.utils import abstractclassmethod


class IntentParser(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def get_slots(self, text, intent=None):
        pass

    @abstractclassmethod
    def from_dict(cls, obj_dict):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.to_dict() == other.to_dict()

    def __ne__(self, other):
        return not self.__eq__(other)
