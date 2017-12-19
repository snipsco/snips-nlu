from abc import ABCMeta, abstractmethod

from snips_nlu.utils import classproperty


class Config(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, obj_dict):
        raise NotImplementedError


class ProcessingUnitConfig(Config):
    __metaclass__ = ABCMeta

    @classproperty
    def unit_name(self):
        raise NotImplementedError
