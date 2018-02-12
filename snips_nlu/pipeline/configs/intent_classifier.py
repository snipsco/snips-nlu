from copy import deepcopy

from snips_nlu.pipeline.configs import Config, ProcessingUnitConfig
from snips_nlu.utils import classproperty


class LogRegIntentClassifierConfig(ProcessingUnitConfig):
    # pylint: disable=line-too-long
    """Configuration of a :class:`.LogRegIntentClassifier`

    Args:
        data_augmentation_config (:class:`IntentClassifierDataAugmentationConfig`):
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
    """

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
    """Configuration of a :class:`.Featurizer` object

    Args:
         sublinear_tf (bool, optional): Whether or not to use sublinear
            (vs linear) term frequencies, default is *False*.
    """

    def __init__(self, sublinear_tf=False):
        self.sublinear_tf = sublinear_tf

    def to_dict(self):
        return {
            "sublinear_tf": self.sublinear_tf
        }

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)
