from __future__ import unicode_literals

from snips_nlu.common.from_dict import FromDict
from snips_nlu.constants import STOP_WORDS
from snips_nlu.pipeline.configs import (
    Config, ProcessingUnitConfig, default_features_factories)
from snips_nlu.resources import merge_required_resources


class CRFSlotFillerConfig(FromDict, ProcessingUnitConfig):
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
        data_augmentation_config (dict or :class:`.SlotFillerDataAugmentationConfig`, optional):
            Specify how to augment data before training the CRF, see the
            corresponding config object for more details.
        random_seed (int, optional): Specify to make the CRF training
            deterministic and reproducible (default=None)
    """

    # pylint: enable=line-too-long

    def __init__(self, feature_factory_configs=None,
                 tagging_scheme=None, crf_args=None,
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
        self._data_augmentation_config = None
        self.data_augmentation_config = data_augmentation_config
        self.random_seed = random_seed

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

    @property
    def unit_name(self):
        from snips_nlu.slot_filler import CRFSlotFiller
        return CRFSlotFiller.unit_name

    def get_required_resources(self):
        # Import here to avoid circular imports
        from snips_nlu.slot_filler.feature_factory import CRFFeatureFactory

        resources = self.data_augmentation_config.get_required_resources()
        for config in self.feature_factory_configs:
            factory = CRFFeatureFactory.from_config(config)
            resources = merge_required_resources(
                resources, factory.get_required_resources())
        return resources

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "feature_factory_configs": self.feature_factory_configs,
            "crf_args": self.crf_args,
            "tagging_scheme": self.tagging_scheme.value,
            "data_augmentation_config":
                self.data_augmentation_config.to_dict(),
            "random_seed": self.random_seed
        }


class SlotFillerDataAugmentationConfig(FromDict, Config):
    """Specify how to augment data before training the CRF

    Data augmentation essentially consists in creating additional utterances
    by combining utterance patterns and slot values

    Args:
        min_utterances (int, optional): Specify the minimum amount of
            utterances to generate per intent (default=200)
        capitalization_ratio (float, optional): If an entity has one or more
            capitalized values, the data augmentation will randomly capitalize
            its values with a ratio of *capitalization_ratio* (default=.2)
        add_builtin_entities_examples (bool, optional): If True, some builtin
            entity examples will be automatically added to the training data.
            Default is True.
    """

    def __init__(self, min_utterances=200, capitalization_ratio=.2,
                 add_builtin_entities_examples=True):
        self.min_utterances = min_utterances
        self.capitalization_ratio = capitalization_ratio
        self.add_builtin_entities_examples = add_builtin_entities_examples

    def get_required_resources(self):
        return {
            STOP_WORDS: True
        }

    def to_dict(self):
        return {
            "min_utterances": self.min_utterances,
            "capitalization_ratio": self.capitalization_ratio,
            "add_builtin_entities_examples": self.add_builtin_entities_examples
        }


def _default_crf_args():
    return {"c1": .1, "c2": .1, "algorithm": "lbfgs"}
