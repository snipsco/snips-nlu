from abc import ABCMeta, abstractmethod

from snips_nlu.utils import instance_from_dict


class IntentParser(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_intent(self, text):
        raise NotImplementedError

    @abstractmethod
    def get_slots(self, text, intent=None):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, obj_dict):
        return instance_from_dict(obj_dict)

    @abstractmethod
    def to_dict(self):
        pass
