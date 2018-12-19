from __future__ import unicode_literals

from snips_nlu.common.from_dict import FromDict
from snips_nlu.constants import CUSTOM_ENTITY_PARSER_USAGE
from snips_nlu.entity_parser import CustomEntityParserUsage
from snips_nlu.pipeline.configs import ProcessingUnitConfig
from snips_nlu.resources import merge_required_resources


class NLUEngineConfig(FromDict, ProcessingUnitConfig):
    """Configuration of a :class:`.SnipsNLUEngine` object

    Args:
        intent_parsers_configs (list): List of intent parser configs
            (:class:`.ProcessingUnitConfig`). The order in the list determines
            the order in which each parser will be called by the nlu engine.
    """

    def __init__(self, intent_parsers_configs=None):
        from snips_nlu.intent_parser import IntentParser

        if intent_parsers_configs is None:
            from snips_nlu.pipeline.configs import (
                ProbabilisticIntentParserConfig,
                DeterministicIntentParserConfig)
            intent_parsers_configs = [
                DeterministicIntentParserConfig(),
                ProbabilisticIntentParserConfig()
            ]
        self.intent_parsers_configs = [
            IntentParser.get_config(conf) for conf in intent_parsers_configs]

    @property
    def unit_name(self):
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
