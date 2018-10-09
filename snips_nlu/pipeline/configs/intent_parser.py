from __future__ import unicode_literals

from copy import deepcopy

from snips_nlu.constants import CUSTOM_ENTITY_PARSER_USAGE
from snips_nlu.entity_parser import CustomEntityParserUsage
from snips_nlu.pipeline.configs import ProcessingUnitConfig
from snips_nlu.pipeline.processing_unit import get_processing_unit_config
from snips_nlu.resources import merge_required_resources
from snips_nlu.utils import classproperty


class ProbabilisticIntentParserConfig(ProcessingUnitConfig):
    """Configuration of a :class:`.ProbabilisticIntentParser` object

    Args:
        intent_classifier_config (:class:`.ProcessingUnitConfig`): The
            configuration of the underlying intent classifier, by default
            it uses a :class:`.LogRegIntentClassifierConfig`
        slot_filler_config (:class:`.ProcessingUnitConfig`): The configuration
            that will be used for the underlying slot fillers, by default it
            uses a :class:`.CRFSlotFillerConfig`
    """

    # pylint: disable=super-init-not-called
    def __init__(self, intent_classifier_config=None, slot_filler_config=None):
        if intent_classifier_config is None:
            from snips_nlu.pipeline.configs import LogRegIntentClassifierConfig
            intent_classifier_config = LogRegIntentClassifierConfig()
        if slot_filler_config is None:
            from snips_nlu.pipeline.configs import CRFSlotFillerConfig
            slot_filler_config = CRFSlotFillerConfig()
        self.intent_classifier_config = get_processing_unit_config(
            intent_classifier_config)
        self.slot_filler_config = get_processing_unit_config(
            slot_filler_config)

    # pylint: enable=super-init-not-called

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.intent_parser import ProbabilisticIntentParser
        return ProbabilisticIntentParser.unit_name

    def get_required_resources(self):
        resources = self.intent_classifier_config.get_required_resources()
        resources = merge_required_resources(
            resources, self.slot_filler_config.get_required_resources())
        return resources

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "slot_filler_config": self.slot_filler_config.to_dict(),
            "intent_classifier_config": self.intent_classifier_config.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)


class DeterministicIntentParserConfig(ProcessingUnitConfig):
    """Configuration of a :class:`.DeterministicIntentParser`

    Args:
        max_queries (int, optional): Maximum number of regex patterns per
            intent. 50 by default.
        max_pattern_length (int, optional): Maximum length of regex patterns.


    This allows to deactivate the usage of regular expression when they are
    too big to avoid explosion in time and memory

    Note:
        In the future, a FST will be used instead of regexps, removing the need
        for all this
    """

    # pylint: disable=super-init-not-called
    def __init__(self, max_queries=100, max_pattern_length=1000):
        self.max_queries = max_queries
        self.max_pattern_length = max_pattern_length

    # pylint: enable=super-init-not-called

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.intent_parser.deterministic_intent_parser import \
            DeterministicIntentParser
        return DeterministicIntentParser.unit_name

    def get_required_resources(self):
        return {
            CUSTOM_ENTITY_PARSER_USAGE: CustomEntityParserUsage.WITHOUT_STEMS
        }

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "max_queries": self.max_queries,
            "max_pattern_length": self.max_pattern_length
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)
