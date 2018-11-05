from __future__ import unicode_literals

from copy import deepcopy

from snips_nlu.constants import (
    CUSTOM_ENTITY_PARSER_USAGE, NOISE, STEMS, STOP_WORDS, WORD_CLUSTERS)
from snips_nlu.entity_parser.custom_entity_parser import \
    CustomEntityParserUsage
from snips_nlu.pipeline.configs import Config, ProcessingUnitConfig
from snips_nlu.resources import merge_required_resources
from snips_nlu.utils import classproperty


class FastTextIntentClassifierConfig(ProcessingUnitConfig):
    """Configuration of a :class:`.Featurizer` object

    Args:
        sublinear_tf (bool, optional): Whether or not to use sublinear
            (vs linear) term frequencies, default is *False*.
        pvalue_threshold (float, optional): max pvalue for a feature to be
        kept in the feature selection
    """

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.intent_classifier.fastext_intent_classifier \
            import FastTextIntentClassifier
        return FastTextIntentClassifier.unit_name

    def __init__(self, embedding_dims=None, learning_rates=None, epochs=None,
                 validation_ratio=0.1, n_threads=4,
                 data_augmentation_config=None, use_stemming=False,
                 unknown_word_prob=0, random_seed=None,
                 unknown_words_replacement_string=None):
        if learning_rates is None:
            learning_rates = [0.1, 0.25, 0.5, 0.75]
        if embedding_dims is None:
            embedding_dims = [3, 5, 7, 10]
        if epochs is None:
            epochs = [10, 20, 30, 50]
        if data_augmentation_config is None:
            data_augmentation_config = IntentClassifierDataAugmentationConfig()
        self.epochs = epochs
        self.embedding_dims = embedding_dims
        self.learning_rates = learning_rates
        self.validation_ratio = validation_ratio
        self.n_threads = n_threads
        self.data_augmentation_config = data_augmentation_config
        self.unknown_word_prob = unknown_word_prob
        self.unknown_words_replacement_string = \
            unknown_words_replacement_string
        self.use_stemming = use_stemming
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

    def get_required_resources(self):
        if self.use_stemming:
            parser_usage = CustomEntityParserUsage.WITH_STEMS
        else:
            parser_usage = CustomEntityParserUsage.WITHOUT_STEMS
        return {
            STEMS: self.use_stemming,
            CUSTOM_ENTITY_PARSER_USAGE: parser_usage
        }

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "embedding_dims": self.embedding_dims,
            "epochs": self.epochs,
            "learning_rates": self.learning_rates,
            "n_threads": self.n_threads,
            "validation_ratio": self.validation_ratio,
            "random_seed": self.random_seed,
            "use_stemming": self.use_stemming,
            "unknown_words_replacement_string": \
                self.unknown_words_replacement_string,
            "data_augmentation_config": self.data_augmentation_config.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)


class LogRegIntentClassifierConfig(ProcessingUnitConfig):
    # pylint: disable=line-too-long
    """Configuration of a :class:`.LogRegIntentClassifier`

    Args:
        data_augmentation_config (
        :class:`IntentClassifierDataAugmentationConfig`):
            Defines the strategy of the underlying data augmentation
        featurizer_config (:class:`FeaturizerConfig`): Configuration of the
            :class:`.Featurizer` used underneath
        random_seed (int, optional): Allows to fix the seed ot have
            reproducible trainings
    """

    # pylint: enable=line-too-long

    # pylint: disable=super-init-not-called
    def __init__(self, data_augmentation_config=None, featurizer_config=None,
                 random_seed=None):
        if data_augmentation_config is None:
            data_augmentation_config = IntentClassifierDataAugmentationConfig()
        if featurizer_config is None:
            featurizer_config = FeaturizerConfig()
        self._data_augmentation_config = None
        self.data_augmentation_config = data_augmentation_config
        self._featurizer_config = None
        self.featurizer_config = featurizer_config
        self.random_seed = random_seed

    # pylint: enable=super-init-not-called

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

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.intent_classifier import LogRegIntentClassifier
        return LogRegIntentClassifier.unit_name

    def get_required_resources(self):
        resources = self.data_augmentation_config.get_required_resources()
        resources = merge_required_resources(
            resources, self.featurizer_config.get_required_resources())
        return resources

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "data_augmentation_config":
                self.data_augmentation_config.to_dict(),
            "featurizer_config": self.featurizer_config.to_dict(),
            "random_seed": self.random_seed
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)


class IntentClassifierDataAugmentationConfig(Config):
    """Configuration used by a :class:`.LogRegIntentClassifier` which defines
        how to augment data to improve the training of the classifier

    Args:
        min_utterances (int, optional): The minimum number of utterances to
            automatically generate for each intent, based on the existing
            utterances. Default is 20.
        noise_factor (int, optional): Defines the size of the noise to
            generate to train the implicit *None* intent, as a multiplier of
            the average size of the other intents. Default is 5.
        add_builtin_entities_examples (bool, optional): If True, some builtin
            entity examples will be automatically added to the training data.
            Default is True.
    """

    def __init__(self, min_utterances=20, noise_factor=5,
                 add_builtin_entities_examples=True, unknown_word_prob=0,
                 unknown_words_replacement_string=None,
                 max_unknown_words=None):
        self.min_utterances = min_utterances
        self.noise_factor = noise_factor
        self.add_builtin_entities_examples = add_builtin_entities_examples
        self.unknown_word_prob = unknown_word_prob
        self.unknown_words_replacement_string = \
            unknown_words_replacement_string
        self.max_unknown_words = max_unknown_words
        if unknown_word_prob > 0 and unknown_words_replacement_string is None:
            raise ValueError("unknown_word_prob is positive (%s) but the "
                             "replacement string is None" % unknown_word_prob)

    @staticmethod
    def get_required_resources():
        return {
            NOISE: True,
            STOP_WORDS: True
        }

    def to_dict(self):
        return {
            "min_utterances": self.min_utterances,
            "noise_factor": self.noise_factor,
            "add_builtin_entities_examples":
                self.add_builtin_entities_examples,
            "unknown_word_prob": self.unknown_word_prob,
            "unknown_words_replacement_string":
                self.unknown_words_replacement_string,
            "max_unknown_words": self.max_unknown_words
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


class FeaturizerConfig(Config):
    """Configuration of a :class:`.Featurizer` object

    Args:
        sublinear_tf (bool, optional): Whether or not to use sublinear
            (vs linear) term frequencies, default is *False*.
        pvalue_threshold (float, optional): max pvalue for a feature to be
        kept in the feature selection
    """

    def __init__(self, sublinear_tf=False, pvalue_threshold=0.4,
                 word_clusters_name=None, use_stemming=False):
        self.sublinear_tf = sublinear_tf
        self.pvalue_threshold = pvalue_threshold
        self.word_clusters_name = word_clusters_name
        self.use_stemming = use_stemming

    def get_required_resources(self):
        if self.use_stemming:
            parser_usage = CustomEntityParserUsage.WITH_STEMS
        else:
            parser_usage = CustomEntityParserUsage.WITHOUT_STEMS
        if self.word_clusters_name is not None:
            word_clusters = {self.word_clusters_name}
        else:
            word_clusters = set()
        return {
            WORD_CLUSTERS: word_clusters,
            STEMS: self.use_stemming,
            CUSTOM_ENTITY_PARSER_USAGE: parser_usage
        }

    def to_dict(self):
        return {
            "sublinear_tf": self.sublinear_tf,
            "pvalue_threshold": self.pvalue_threshold,
            "word_clusters_name": self.word_clusters_name,
            "use_stemming": self.use_stemming
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)
