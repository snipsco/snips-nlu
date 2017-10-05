# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from copy import deepcopy

from snips_nlu.utils import namedtuple_with_defaults


class Config(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_dict(self):
        pass


class NamedTupleConfigMixin(Config):
    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, Config) else v
            for k, v in self._asdict()
        }


_DataAugmentationConfig = namedtuple_with_defaults(
    '_DataAugmentationConfig',
    'max_utterances',
    {
        'max_utterances': 200
    }
)


class DataAugmentationConfig(_DataAugmentationConfig, NamedTupleConfigMixin):
    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


_IntentClassificationConfig = namedtuple_with_defaults(
    '_IntentClassificationConfig',
    'data_augmentation_config',
    {
        'data_augmentation_config': DataAugmentationConfig()
    }
)
