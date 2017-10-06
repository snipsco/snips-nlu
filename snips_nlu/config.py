# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from copy import deepcopy

from snips_nlu.utils import namedtuple_with_defaults, abstractclassmethod


class Config(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_dict(self):
        pass

    @abstractclassmethod
    def from_dict(cls, obj_dict):
        pass


class FromDictMixin(object):
    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class NamedTupleConfigMixin(Config):
    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, Config) else v
            for k, v in self._asdict().iteritems()
        }


_DataAugmentationConfig = namedtuple_with_defaults(
    '_DataAugmentationConfig',
    'min_utterances capitalization_ratio',
    {
        'min_utterances': 200,
        'capitalization_ratio': .2
    }
)


class DataAugmentationConfig(_DataAugmentationConfig, NamedTupleConfigMixin,
                             FromDictMixin):
    pass


_IntentClassifierConfig = namedtuple_with_defaults(
    '_IntentClassifierConfig',
    'data_augmentation_config noise_factor log_reg_args',
    {
        'data_augmentation_config': DataAugmentationConfig(min_utterances=20),
        'noise_factor': 5,
        'log_reg_args':
            {
                "loss": 'log',
                "penalty": 'l2',
                "class_weight": 'balanced',
                "n_iter": 5,
                "random_state": 42,
                "n_jobs": -1
            }
    }
)


class IntentClassifierConfig(_IntentClassifierConfig,
                             NamedTupleConfigMixin):
    @classmethod
    def from_dict(cls, obj_dict):
        args = deepcopy(obj_dict)
        args["data_augmentation_config"] = DataAugmentationConfig.from_dict(
            args["data_augmentation_config"])
        return cls(**args)


_CRFFeaturesConfig = namedtuple_with_defaults(
    "_CRFFeaturesConfig", "base_drop_ratio entities_offsets",
    {
        "base_drop_ratio": .5,
        "entities_offsets": [-2, -1, 0]
    }
)


class CRFFeaturesConfig(_CRFFeaturesConfig, NamedTupleConfigMixin,
                        FromDictMixin):
    pass


_ProbabilisticIntentParserConfig = namedtuple_with_defaults(
    'ProbabilisticIntentParserConfig',
    'data_augmentation_config crf_features_config',
    {
        'data_augmentation_config': DataAugmentationConfig(),
        'crf_features_config': CRFFeaturesConfig()
    }
)


class ProbabilisticIntentParserConfig(_ProbabilisticIntentParserConfig,
                                      NamedTupleConfigMixin):
    @classmethod
    def from_dict(cls, obj_dict):
        args = deepcopy(obj_dict)
        args["data_augmentation_config"] = DataAugmentationConfig.from_dict(
            args["data_augmentation_config"])
        return cls(**args)


_RegexTrainingConfig = namedtuple_with_defaults(
    '_RegexTrainingConfig',
    'max_queries max_entities',
    {
        'max_queries': 50,
        'max_entities': 200
    }
)


class RegexTrainingConfig(_RegexTrainingConfig, NamedTupleConfigMixin,
                          FromDictMixin):
    pass


_NLUConfig = namedtuple_with_defaults(
    '_NLUConfig',
    'intent_classifier_config probabilistic_intent_parser_config'
    ' regex_training_config',
    {
        'intent_classifier_config': IntentClassifierConfig(),
        'probabilistic_intent_parser_config':
            ProbabilisticIntentParserConfig(),
        'regex_training_config': RegexTrainingConfig()
    }
)


class NLUConfig(_NLUConfig, NamedTupleConfigMixin):
    @classmethod
    def from_dict(cls, obj_dict):
        args = deepcopy(obj_dict)
        args["intent_classifier_config"] = IntentClassifierConfig \
            .from_dict(args["intent_classifier_config"])
        args["probabilistic_intent_parser_config"] = \
            ProbabilisticIntentParserConfig.from_dict(
                args["probabilistic_intent_parser_config"])
        return cls(**args)
