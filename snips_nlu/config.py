# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from copy import deepcopy

from snips_nlu.utils import namedtuple_with_defaults, abstractclassmethod


class Config(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @abstractclassmethod
    def from_dict(cls, obj_dict):
        raise NotImplementedError


class NamedTupleConfigMixin(Config):
    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, Config) else v
            for k, v in self._asdict().iteritems()
        }


_IntentClassifierDataAugmentationConfig = namedtuple_with_defaults(
    '_IntentClassifierDataAugmentationConfig',
    'min_utterances noise_factor unknown_word_prob '
    'unknown_words_replacement_string',
    {
        'min_utterances': 20,
        'noise_factor': 5,
        'unknown_word_prob': 0,
        'unknown_words_replacement_string': None
    }
)


class IntentClassifierDataAugmentationConfig(
        _IntentClassifierDataAugmentationConfig,
        NamedTupleConfigMixin):
    pass


_SlotFillerDataAugmentationConfig = namedtuple_with_defaults(
    'SlotFillerDataAugmentationConfig',
    'min_utterances capitalization_ratio',
    {
        'min_utterances': 200,
        'capitalization_ratio': .2
    }
)


class SlotFillerDataAugmentationConfig(_SlotFillerDataAugmentationConfig,
                                       NamedTupleConfigMixin):
    pass


_FeaturizerConfig = namedtuple_with_defaults(
    '_FeaturizerConfig',
    'sublinear_tf',
    {'sublinear_tf': False}
)


class FeaturizerConfig(_FeaturizerConfig, NamedTupleConfigMixin):
    pass


_IntentClassifierConfig = namedtuple_with_defaults(
    '_IntentClassifierConfig',
    'data_augmentation_config featurizer_config log_reg_args',
    {
        'data_augmentation_config':
            IntentClassifierDataAugmentationConfig(
                min_utterances=20, noise_factor=5),
        'log_reg_args':
            {
                "loss": 'log',
                "penalty": 'l2',
                "class_weight": 'balanced',
                "n_iter": 5,
                "random_state": 42,
                "n_jobs": -1
            },
        'featurizer_config': FeaturizerConfig()
    }
)


class IntentClassifierConfig(_IntentClassifierConfig,
                             NamedTupleConfigMixin):
    @classmethod
    def from_dict(cls, obj_dict):
        args = deepcopy(obj_dict)
        args["featurizer_config"] = FeaturizerConfig.from_dict(
            args["featurizer_config"])
        args["data_augmentation_config"] = \
            IntentClassifierDataAugmentationConfig.from_dict(
                args["data_augmentation_config"])
        return cls(**args)


_CRFFeaturesConfig = namedtuple_with_defaults(
    "_CRFFeaturesConfig", "base_drop_ratio entities_offsets",
    {
        "base_drop_ratio": .5,
        "entities_offsets": [-2, -1, 0]
    }
)


class CRFFeaturesConfig(_CRFFeaturesConfig, NamedTupleConfigMixin):
    pass


_ProbabilisticIntentParserConfig = namedtuple_with_defaults(
    'ProbabilisticIntentParserConfig',
    'data_augmentation_config crf_features_config',
    {
        'data_augmentation_config': SlotFillerDataAugmentationConfig(),
        'crf_features_config': CRFFeaturesConfig()
    }
)


class ProbabilisticIntentParserConfig(_ProbabilisticIntentParserConfig,
                                      NamedTupleConfigMixin):
    @classmethod
    def from_dict(cls, obj_dict):
        args = deepcopy(obj_dict)
        args["crf_features_config"] = CRFFeaturesConfig.from_dict(
            args["crf_features_config"])
        args["data_augmentation_config"] = \
            SlotFillerDataAugmentationConfig.from_dict(
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


class RegexTrainingConfig(_RegexTrainingConfig, NamedTupleConfigMixin):
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
        args['regex_training_config'] = RegexTrainingConfig \
            .from_dict(args['regex_training_config'])
        args["intent_classifier_config"] = IntentClassifierConfig \
            .from_dict(args["intent_classifier_config"])
        args["probabilistic_intent_parser_config"] = \
            ProbabilisticIntentParserConfig.from_dict(
                args["probabilistic_intent_parser_config"])
        return cls(**args)
