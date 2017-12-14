from snips_nlu.configs.config import Serializable
from snips_nlu.configs.features import default_features_factories
from snips_nlu.slot_filler.crf_utils import TaggingScheme


class SlotFillerDataAugmentationConfig(Serializable):
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


def _default_entities_offsets():
    return [-2, -1, 0]


class CRFSlotFillerConfig(Serializable):
    def __init__(self, feature_factory_configs=None,
                 tagging_scheme=TaggingScheme.BIO.value, crf_args=None,
                 entities_offsets=None,
                 exhaustive_permutations_threshold=4 ** 3,
                 data_augmentation_config=None, random_seed=None):
        if feature_factory_configs is None:
            feature_factory_configs = default_features_factories()
        if crf_args is None:
            crf_args = _default_crf_args()
        if entities_offsets is None:
            entities_offsets = _default_entities_offsets()
        if data_augmentation_config is None:
            data_augmentation_config = SlotFillerDataAugmentationConfig()
        self.feature_factory_configs = feature_factory_configs
        self._tagging_scheme = None
        self.tagging_scheme = tagging_scheme
        self.crf_args = crf_args
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
            "feature_factory_configs": self.feature_factory_configs,
            "crf_args": self.crf_args,
            "tagging_scheme": self.tagging_scheme.value,
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
