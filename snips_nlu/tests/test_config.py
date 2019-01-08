# coding=utf-8
from __future__ import unicode_literals

import io
import unittest

from snips_nlu_ontology import get_all_languages

from snips_nlu import SnipsNLUEngine
from snips_nlu.constants import LANGUAGE, RES_INTENT, RES_INTENT_NAME
from snips_nlu.dataset import Dataset
from snips_nlu.default_configs import DEFAULT_CONFIGS
from snips_nlu.intent_classifier import LogRegIntentClassifier
from snips_nlu.pipeline.configs import (
    CRFSlotFillerConfig, DeterministicIntentParserConfig, FeaturizerConfig,
    IntentClassifierDataAugmentationConfig, LogRegIntentClassifierConfig,
    NLUEngineConfig, ProbabilisticIntentParserConfig,
    SlotFillerDataAugmentationConfig)
from snips_nlu.tests.utils import SnipsTest


class TestConfig(SnipsTest):
    def test_intent_classifier_data_augmentation_config(self):
        # Given
        config_dict = {
            "min_utterances": 3,
            "noise_factor": 2,
            "add_builtin_entities_examples": False,
            "unknown_word_prob": 0.1,
            "unknown_words_replacement_string": "foobar",
            "max_unknown_words": None,
        }

        # When
        config = IntentClassifierDataAugmentationConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_slot_filler_data_augmentation_config(self):
        # Given
        config_dict = {
            "min_utterances": 42,
            "capitalization_ratio": 0.66,
            "add_builtin_entities_examples": False
        }

        # When
        config = SlotFillerDataAugmentationConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_featurizer_config(self):
        # Given
        config_dict = {
            "sublinear_tf": True,
            "pvalue_threshold": 0.4,
            "word_clusters_name": None,
            "use_stemming": False
        }

        # When
        config = FeaturizerConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_intent_classifier_config(self):
        # Given
        config_dict = {
            "unit_name": LogRegIntentClassifier.unit_name,
            "data_augmentation_config":
                IntentClassifierDataAugmentationConfig().to_dict(),
            "featurizer_config": FeaturizerConfig().to_dict(),
            "random_seed": 42
        }

        # When
        config = LogRegIntentClassifierConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_crf_slot_filler_config(self):
        # Given
        feature_factories = [
            {
                "args": {
                    "common_words_gazetteer_name": None,
                    "use_stemming": True,
                    "n": 1
                },
                "factory_name": "get_ngram_fn",
                "offsets": [-2, -1, 0, 1, 2]
            },
            {
                "args": {},
                "factory_name": "is_digit",
                "offsets": [-1, 0, 1]
            }
        ]
        config_dict = {
            "unit_name": "crf_slot_filler",
            "feature_factory_configs": feature_factories,
            "tagging_scheme": 2,
            "crf_args": {
                "c1": .2,
                "c2": .3,
                "algorithm": "lbfgs"
            },
            "data_augmentation_config":
                SlotFillerDataAugmentationConfig().to_dict(),
            "random_seed": 43
        }

        # When
        config = CRFSlotFillerConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_probabilistic_intent_parser_config(self):
        # Given
        config_dict = {
            "unit_name": "probabilistic_intent_parser",
            "intent_classifier_config":
                LogRegIntentClassifierConfig().to_dict(),
            "slot_filler_config": CRFSlotFillerConfig().to_dict(),
        }

        # When
        config = ProbabilisticIntentParserConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_deterministic_parser_config(self):
        # Given
        config_dict = {
            "unit_name": "deterministic_intent_parser",
            "max_queries": 666,
            "max_pattern_length": 333,
            "ignore_stop_words": True
        }

        # When
        config = DeterministicIntentParserConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_nlu_config_from_dict(self):
        # Given
        config_dict = {
            "unit_name": "nlu_engine",
            "intent_parsers_configs": [
                DeterministicIntentParserConfig().to_dict(),
                ProbabilisticIntentParserConfig().to_dict()
            ]
        }

        # When
        config = NLUEngineConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_default_configs_should_work(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: TurnLightOn
utterances:
- turn on the lights
- please switch on the light
- switch the light on
- can you turn the light on ?
- I need you to turn on the lights

---
type: intent
name: GetWeather
utterances:
- what is the weather today
- What's the weather in tokyo today?
- Can you tell me the weather please ?
- what is the weather forecast for this weekend""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        for language in get_all_languages():
            # When
            config = DEFAULT_CONFIGS.get(language)
            self.assertIsNotNone(config, "Missing default config for '%s'"
                                 % language)
            dataset[LANGUAGE] = language
            engine = SnipsNLUEngine(config).fit(dataset)
            result = engine.parse("Please give me the weather in Paris")

            # Then
            self.assertIsNotNone(result[RES_INTENT])
            intent_name = result[RES_INTENT][RES_INTENT_NAME]
            self.assertEqual("GetWeather", intent_name)


if __name__ == '__main__':
    unittest.main()
