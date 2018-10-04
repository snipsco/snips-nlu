from __future__ import unicode_literals

from builtins import map
from copy import deepcopy

from snips_nlu.constants import CUSTOM_ENTITY_PARSER_USAGE
from snips_nlu.entity_parser import CustomEntityParserUsage
from snips_nlu.pipeline.configs import ProcessingUnitConfig
from snips_nlu.pipeline.processing_unit import get_processing_unit_config
from snips_nlu.resources import merge_required_resources
from snips_nlu.utils import classproperty


class NLUEngineConfig(ProcessingUnitConfig):
    """Configuration of a :class:`.SnipsNLUEngine` object

    Args:
        intent_parsers_configs (list): List of intent parser configs
            (:class:`.ProcessingUnitConfig`). The order in the list determines
            the order in which each parser will be called by the nlu engine.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, intent_parsers_configs=None):

        if intent_parsers_configs is None:
            from snips_nlu.pipeline.configs import (
                ProbabilisticIntentParserConfig,
                DeterministicIntentParserConfig)
            intent_parsers_configs = [
                DeterministicIntentParserConfig(),
                ProbabilisticIntentParserConfig()
            ]
        self.intent_parsers_configs = list(map(get_processing_unit_config,
                                               intent_parsers_configs))

    # pylint: enable=super-init-not-called

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        from snips_nlu.nlu_engine.nlu_engine import SnipsNLUEngine
        return SnipsNLUEngine.unit_name

    def get_required_resources(self):
        # Resolving custom slot values must be done without stemming
        resources = {
            CUSTOM_ENTITY_PARSER_USAGE: CustomEntityParserUsage.WITHOUT_STEMS
        }
        for config in self.intent_parsers_configs:
            resources = merge_required_resources(
                resources, config.get_required_resources())
        return resources

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "intent_parsers_configs": [
                config.to_dict() for config in self.intent_parsers_configs
            ]
        }

    @classmethod
    def from_dict(cls, obj_dict):
        d = obj_dict
        if "unit_name" in obj_dict:
            d = deepcopy(obj_dict)
            d.pop("unit_name")
        return cls(**d)
