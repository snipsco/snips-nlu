# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod

from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.utils import abstractclassmethod


class Config(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @abstractclassmethod
    def from_dict(cls, obj_dict):
        raise NotImplementedError


class IntentClassifierDataAugmentationConfig(Config):
    def __init__(self, min_utterances=20, noise_factor=5, unknown_word_prob=0,
                 unknown_words_replacement_string=None):
        self.min_utterances = min_utterances
        self.noise_factor = noise_factor
        self.unknown_word_prob = unknown_word_prob
        self.unknown_words_replacement_string = \
            unknown_words_replacement_string

    def to_dict(self):
        return {
            "min_utterances": self.min_utterances,
            "noise_factor": self.noise_factor,
            "unknown_word_prob": self.unknown_word_prob,
            "unknown_words_replacement_string":
                self.unknown_words_replacement_string,
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class FeaturizerConfig(Config):
    def __init__(self, sublinear_tf=False):
        self.sublinear_tf = sublinear_tf

    def to_dict(self):
        return {
            "sublinear_tf": self.sublinear_tf
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class IntentClassifierConfig(Config):
    def __init__(
            self,
            data_augmentation_config=IntentClassifierDataAugmentationConfig(),
            log_reg_args=None, featurizer_config=FeaturizerConfig(),
            random_seed=None):
        self._data_augmentation_config = None
        self.data_augmentation_config = data_augmentation_config
        if log_reg_args is None:
            log_reg_args = {
                "loss": "log",
                "penalty": "l2",
                "class_weight": "balanced",
                "n_iter": 5,
                "n_jobs": -1
            }
        self.log_reg_args = log_reg_args
        self._featurizer_config = None
        self.featurizer_config = featurizer_config
        self.random_seed = random_seed

    @property
    def data_augmentation_config(self):
        return self._data_augmentation_config

    @data_augmentation_config.setter
    def data_augmentation_config(self, value):
        if isinstance(value, dict):
            self._data_augmentation_config = \
                IntentClassifierDataAugmentationConfig.from_dict(value)
        elif isinstance(value, IntentClassifierDataAugmentationConfig):
            self._data_augmentation_config = value
        else:
            raise TypeError("Expected instance of "
                            "IntentClassifierDataAugmentationConfig or dict"
                            "but received: %s" % type(value))

    @property
    def featurizer_config(self):
        return self._featurizer_config

    @featurizer_config.setter
    def featurizer_config(self, value):
        if isinstance(value, dict):
            self._featurizer_config = \
                FeaturizerConfig.from_dict(value)
        elif isinstance(value, FeaturizerConfig):
            self._featurizer_config = value
        else:
            raise TypeError("Expected instance of FeaturizerConfig or dict"
                            "but received: %s" % type(value))

    def to_dict(self):
        return {
            "data_augmentation_config":
                self.data_augmentation_config.to_dict(),
            "log_reg_args": self.log_reg_args,
            "featurizer_config": self.featurizer_config.to_dict(),
            "random_seed": self.random_seed
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class SlotFillerDataAugmentationConfig(Config):
    def __init__(self, min_utterances=200, capitalization_ratio=.2):
        self.min_utterances = min_utterances
        self.capitalization_ratio = capitalization_ratio

    def to_dict(self):
        return {
            "min_utterances": self.min_utterances,
            "capitalization_ratio": self.capitalization_ratio
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class CRFSlotFillerConfig(Config):
    def __init__(self, tagging_scheme=TaggingScheme.BIO.value, crf_args=None,
                 features_drop_out=None, entities_offsets=None,
                 exhaustive_permutations_threshold=4 ** 3,
                 data_augmentation_config=SlotFillerDataAugmentationConfig(),
                 random_seed=None):
        if crf_args is None:
            crf_args = {
                "c1": .1,
                "c2": .1,
                "algorithm": "lbfgs"
            }
        if features_drop_out is None:
            features_drop_out = {
                "collection_match": 0.5
            }
        if entities_offsets is None:
            entities_offsets = [-2, -1, 0]
        self._tagging_scheme = None
        self.tagging_scheme = tagging_scheme
        self.crf_args = crf_args
        self.features_drop_out = features_drop_out
        self.entities_offsets = entities_offsets
        self.exhaustive_permutations_threshold = \
            exhaustive_permutations_threshold
        self._data_augmentation_config = None
        self.data_augmentation_config = data_augmentation_config
        self.random_seed = random_seed

    @property
    def tagging_scheme(self):
        return self._tagging_scheme

    @tagging_scheme.setter
    def tagging_scheme(self, value):
        if isinstance(value, TaggingScheme):
            self._tagging_scheme = value
        elif isinstance(value, int):
            self._tagging_scheme = TaggingScheme(value)
        else:
            raise TypeError("Expected instance of TaggingScheme or int but"
                            "received: %s" % type(value))

    @property
    def data_augmentation_config(self):
        return self._data_augmentation_config

    @data_augmentation_config.setter
    def data_augmentation_config(self, value):
        if isinstance(value, dict):
            self._data_augmentation_config = \
                SlotFillerDataAugmentationConfig.from_dict(value)
        elif isinstance(value, SlotFillerDataAugmentationConfig):
            self._data_augmentation_config = value
        else:
            raise TypeError("Expected instance of "
                            "SlotFillerDataAugmentationConfig or dict but "
                            "received: %s" % type(value))

    def to_dict(self):
        return {
            "crf_args": self.crf_args,
            "tagging_scheme": self.tagging_scheme.value,
            "features_drop_out": self.features_drop_out,
            "entities_offsets": self.entities_offsets,
            "exhaustive_permutations_threshold":
                self.exhaustive_permutations_threshold,
            "data_augmentation_config":
                self.data_augmentation_config.to_dict(),
            "random_seed": self.random_seed
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class ProbabilisticIntentParserConfig(Config):
    def __init__(self, intent_classifier_config=IntentClassifierConfig(),
                 crf_slot_filler_config=CRFSlotFillerConfig()):
        self._intent_classifier_config = None
        self.intent_classifier_config = intent_classifier_config
        self._crf_slot_filler_config = None
        self.crf_slot_filler_config = crf_slot_filler_config

    @property
    def crf_slot_filler_config(self):
        return self._crf_slot_filler_config

    @crf_slot_filler_config.setter
    def crf_slot_filler_config(self, value):
        if isinstance(value, dict):
            self._crf_slot_filler_config = \
                CRFSlotFillerConfig.from_dict(value)
        elif isinstance(value, CRFSlotFillerConfig):
            self._crf_slot_filler_config = value
        else:
            raise TypeError("Expected instance of CRFSlotFillerConfig or dict "
                            "but received: %s" % type(value))

    @property
    def intent_classifier_config(self):
        return self._intent_classifier_config

    @intent_classifier_config.setter
    def intent_classifier_config(self, value):
        if isinstance(value, dict):
            self._intent_classifier_config = \
                IntentClassifierConfig.from_dict(value)
        elif isinstance(value, IntentClassifierConfig):
            self._intent_classifier_config = value
        else:
            raise TypeError("Expected instance of IntentClassifierConfig or "
                            "dict but received: %s" % type(value))

    def to_dict(self):
        return {
            "crf_slot_filler_config": self.crf_slot_filler_config.to_dict(),
            "intent_classifier_config": self.intent_classifier_config.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class RegexTrainingConfig(Config):
    def __init__(self, max_queries=50, max_entities=200):
        self.max_queries = max_queries
        self.max_entities = max_entities

    def to_dict(self):
        return {
            "max_queries": self.max_queries,
            "max_entities": self.max_entities
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class NLUConfig(Config):
    def __init__(self, intent_classifier_config=IntentClassifierConfig(),
                 probabilistic_intent_parser_config=
                 ProbabilisticIntentParserConfig(),
                 regex_training_config=RegexTrainingConfig()):
        self._intent_classifier_config = None
        self.intent_classifier_config = intent_classifier_config
        self._probabilistic_intent_parser_config = None
        self.probabilistic_intent_parser_config = \
            probabilistic_intent_parser_config
        self._regex_training_config = None
        self.regex_training_config = regex_training_config

    @property
    def intent_classifier_config(self):
        return self._intent_classifier_config

    @intent_classifier_config.setter
    def intent_classifier_config(self, value):
        if isinstance(value, dict):
            self._intent_classifier_config = \
                IntentClassifierConfig.from_dict(value)
        elif isinstance(value, IntentClassifierConfig):
            self._intent_classifier_config = value
        else:
            raise TypeError("Expected instance of IntentClassifierConfig or "
                            "dict but received: %s" % type(value))

    @property
    def probabilistic_intent_parser_config(self):
        return self._probabilistic_intent_parser_config

    @probabilistic_intent_parser_config.setter
    def probabilistic_intent_parser_config(self, value):
        if isinstance(value, dict):
            self._probabilistic_intent_parser_config = \
                ProbabilisticIntentParserConfig.from_dict(value)
        elif isinstance(value, ProbabilisticIntentParserConfig):
            self._probabilistic_intent_parser_config = value
        else:
            raise TypeError("Expected instance of "
                            "ProbabilisticIntentParserConfig or dict but "
                            "received: %s" % type(value))

    @property
    def regex_training_config(self):
        return self._regex_training_config

    @regex_training_config.setter
    def regex_training_config(self, value):
        if isinstance(value, dict):
            self._regex_training_config = \
                RegexTrainingConfig.from_dict(value)
        elif isinstance(value, RegexTrainingConfig):
            self._regex_training_config = value
        else:
            raise TypeError("Expected instance of RegexTrainingConfig or dict "
                            "but received: %s" % type(value))

    def to_dict(self):
        return {
            "intent_classifier_config":
                self.intent_classifier_config.to_dict(),
            "probabilistic_intent_parser_config":
                self.probabilistic_intent_parser_config.to_dict(),
            "regex_training_config": self.regex_training_config.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)
