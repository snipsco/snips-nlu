from abc import ABCMeta, abstractmethod

from snips_nlu.utils import abstractclassmethod


class Serializable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @abstractclassmethod
    def from_dict(cls, obj_dict):
        raise NotImplementedError
