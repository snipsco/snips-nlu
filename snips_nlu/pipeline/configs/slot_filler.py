from __future__ import unicode_literals

from copy import deepcopy

from snips_nlu.pipeline.configs import Config, ProcessingUnitConfig
from snips_nlu.pipeline.configs import default_features_factories
from snips_nlu.utils import classproperty


class CRFSlotFillerConfig(ProcessingUnitConfig):
    # pylint: disable=line-too-long
    """Configuration of a :class:`.CRFSlotFiller`

    Args:
        feature_factory_configs (list, optional): List of configurations that
            specify the list of :class:`.CRFFeatureFactory` to use with the CRF
        tagging_scheme (:class:`.TaggingScheme`, optional): Tagging scheme to
            use to enrich CRF labels (default=BIO)
        crf_args (dict, optional): Allow to overwrite the parameters of the CRF
            defined in *sklearn_crfsuite*, see :class:`sklearn_crfsuite.CRF`
            (default={"c1": .1, "c2": .1, "algorithm": "lbfgs"})
        exhaustive_permutations_threshold (int, optional):
            TODO: properly document this
        data_augmentation_config (dict or :class:`.SlotFillerDataAugmentationConfig`, optional):
            Specify how to augment data before training the CRF, see the
            corresponding config object for more details.
        random_seed (int, optional): Specify to make the CRF training
            deterministic and reproducible (default=None)
    """

    # pylint: enable=line-too-long

    # pylint: disable=super-init-not-called
    def __init__(self, feature_factory_configs=None,
                 tagging_scheme=None, crf_args=None,
                 exhaustive_permutations_threshold=4 ** 3,
                 data_augmentation_config=None, random_seed=None):
        if tagging_scheme is None:
            from snips_nlu.slot_filler.crf_utils import TaggingScheme
            tagging_scheme = TaggingScheme.BIO
        if feature_factory_configs is None:
            feature_factory_configs = default_features_factories()
        if crf_args is None:
            crf_args = _default_crf_args()
        if data_augmentation_config is None:
            data_augmentation_config = SlotFillerDataAugmentationConfig()
        self.feature_factory_configs = feature_factory_configs
        self._tagging_scheme = None
        self.tagging_scheme = tagging_scheme
        self.crf_args = crf_args
        self.exhaustive_permutations_threshold = \
            exhaustive_permutations_threshold
        self._data_augmentation_config = None
        self.data_augmentation_config = data_augmentation_config
        self.random_seed = random_seed

    # pylint: enable=super-init-not-called

    @property
    def tagging_scheme(self):
        return self._tagging_scheme

    @tagging_scheme.setter
    def tagging_scheme(self, value):
        from snips_nlu.slot_filler.crf_utils import TaggingScheme
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

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.slot_filler import CRFSlotFiller
        return CRFSlotFiller.unit_name

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "feature_factory_configs": self.feature_factory_configs,
            "crf_args": self.crf_args,
            "tagging_scheme": self.tagging_scheme.value,
            "exhaustive_permutations_threshold":
                self.exhaustive_permutations_threshold,
            "data_augmentation_config":
                self.data_augmentation_config.to_dict(),
            "random_seed": self.random_seed
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)


class SlotFillerDataAugmentationConfig(Config):
    """Specify how to augment data before training the CRF

    Data augmentation essentially consists in creating additional utterances
    by combining utterance patterns and slot values

    Args:
        min_utterances (int, optional): Specify the minimum amount of
            utterances to generate per intent (default=200)
        capitalization_ratio (float, optional): If an entity has one or more
            capitalized values, the data augmentation will randomly capitalize
            its values with a ratio of *capitalization_ratio* (default=.2)
    """

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


def _default_crf_args():
    return {"c1": .1, "c2": .1, "algorithm": "lbfgs"}
