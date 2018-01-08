# coding=utf-8
import unittest

from snips_nlu.config import (NLUConfig,
                              IntentClassifierDataAugmentationConfig,
                              SlotFillerDataAugmentationConfig,
                              FeaturizerConfig, IntentClassifierConfig,
                              CRFSlotFillerConfig,
                              ProbabilisticIntentParserConfig,
                              DeterministicIntentParserConfig)


class TestConfig(unittest.TestCase):
    def test_intent_classifier_data_augmentation_config(self):
        # Given
        config_dict = {
            "min_utterances": 3,
            "noise_factor": 2,
            "unknown_word_prob": 0.1,
            "unknown_words_replacement_string": "foobar",
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
            "capitalization_ratio": 0.66
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
        }

        # When
        config = FeaturizerConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_intent_classifier_config(self):
        # Given
        config_dict = {
            "data_augmentation_config":
                IntentClassifierDataAugmentationConfig().to_dict(),
            "featurizer_config": FeaturizerConfig().to_dict(),
            "random_seed": 42
        }

        # When
        config = IntentClassifierConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_crf_slot_filler_config(self):
        # Given
        config_dict = {
            "tagging_scheme": 2,
            "crf_args": {
                "c1": .2,
                "c2": .3,
                "algorithm": "lbfgs"
            },
            "features_drop_out": {
                "feature_1": 0.5,
                "feature_2": 0.1
            },
            "entities_offsets": [-2, 0, 3],
            "exhaustive_permutations_threshold": 42,
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
            "intent_classifier_config": IntentClassifierConfig().to_dict(),
            "crf_slot_filler_config": CRFSlotFillerConfig().to_dict(),
        }

        # When
        config = ProbabilisticIntentParserConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_regex_training_config(self):
        # Given
        config_dict = {
            "max_queries": 666,
            "max_entities": 333
        }

        # When
        config = DeterministicIntentParserConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)

    def test_nlu_config_from_dict(self):
        # Given
        config_dict = {
            "probabilistic_intent_parser_config":
                ProbabilisticIntentParserConfig().to_dict(),
            "deterministic_intent_parser_config":
                DeterministicIntentParserConfig().to_dict()
        }

        # When
        config = NLUConfig.from_dict(config_dict)
        serialized_config = config.to_dict()

        # Then
        self.assertDictEqual(config_dict, serialized_config)


if __name__ == '__main__':
    unittest.main()
