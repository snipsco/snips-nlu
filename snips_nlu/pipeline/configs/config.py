from abc import ABCMeta, abstractmethod
from builtins import object

from future.utils import with_metaclass

from snips_nlu.utils import classproperty


class Config(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, obj_dict):
        raise NotImplementedError


class ProcessingUnitConfig(with_metaclass(ABCMeta, Config)):
    @classproperty
    def unit_name(self):
        raise NotImplementedError
