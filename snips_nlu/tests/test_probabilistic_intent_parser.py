from __future__ import unicode_literals

import unittest

from mock import MagicMock, patch, call

from snips_nlu.configs.intent_classifier import IntentClassifierConfig
from snips_nlu.configs.intent_parser import ProbabilisticIntentParserConfig
from snips_nlu.configs.slot_filler import CRFSlotFillerConfig
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.languages import Language
from snips_nlu.tests.utils import BEVERAGE_DATASET


class TestProbabilisticIntentParser(unittest.TestCase):
    def test_should_not_allow_to_fit_with_missing_intents(self):
        # Given
        intent = "MakeTea"
        parser = ProbabilisticIntentParser()
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)

        # When / Then
        expected_missing_intent = "MakeCoffee"
        with self.assertRaises(ValueError) as context:
            parser.fit(dataset, {intent})
        self.assertTrue("These intents must be trained: set([u'%s'])"
                        % expected_missing_intent in context.exception)

    def test_should_be_serializable_before_fitting(self):
        # Given
        parser_config = ProbabilisticIntentParserConfig()
        parser = ProbabilisticIntentParser(parser_config)

        # When
        actual_parser_dict = parser.to_dict()

        # Then
        expected_parser_dict = {
            "config": {
                "crf_slot_filler_config": CRFSlotFillerConfig().to_dict(),
                "intent_classifier_config": IntentClassifierConfig().to_dict()
            },
            "intent_classifier": None,
            "language_code": None,
            "slot_fillers": None,
        }
        self.assertDictEqual(actual_parser_dict, expected_parser_dict)

    def test_should_be_deserializable_before_fitting(self):
        # When
        config = ProbabilisticIntentParserConfig().to_dict()
        parser_dict = {
            "config": config,
            "intent_classifier": None,
            "language_code": None,
            "slot_fillers": None,
        }

        # When
        parser = ProbabilisticIntentParser.from_dict(parser_dict)

        # Then
        self.assertEqual(parser.config.to_dict(), config)
        self.assertIsNone(parser.language)
        self.assertIsNone(parser.intent_classifier)
        self.assertIsNone(parser.slot_fillers)

    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.CRFSlotFiller.fit')
    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.CRFSlotFiller.to_dict')
    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.SnipsIntentClassifier.fit')
    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.SnipsIntentClassifier.to_dict')
    def test_should_be_serializable(self, mock_classifier_to_dict,
                                    mock_classifier_fit,
                                    mock_slot_filler_to_dict,
                                    mock_slot_filler_fit):
        # Given
        mock_classifier_fit.return_value = None
        mock_classifier_to_dict.return_value = {
            "mocked_classifier_key": "mocked_classifier_value"
        }
        mock_slot_filler_fit.return_value = None
        mock_slot_filler_to_dict.return_value = {
            "mocked_slot_filler_key": "mocked_slot_filler_value"
        }

        parser_config = ProbabilisticIntentParserConfig()
        parser = ProbabilisticIntentParser(parser_config)
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        parser.fit(dataset)

        # When
        actual_parser_dict = parser.to_dict()

        # Then
        expected_parser_dict = {
            "config": parser_config.to_dict(),
            "intent_classifier": {
                "mocked_classifier_key": "mocked_classifier_value"
            },
            "language_code": "en",
            "slot_fillers": {
                "MakeCoffee": {
                    "mocked_slot_filler_key": "mocked_slot_filler_value"
                },
                "MakeTea": {
                    "mocked_slot_filler_key": "mocked_slot_filler_value"
                }
            },
        }
        self.assertDictEqual(actual_parser_dict, expected_parser_dict)

    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.SnipsIntentClassifier.from_dict')
    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.CRFSlotFiller')
    def test_should_be_deserializable(self, mock_slot_filler,
                                      mock_classifier_from_dict):
        # When
        language = Language.EN
        mocked_slot_filler = MagicMock()
        mock_slot_filler.from_dict.return_value = mocked_slot_filler
        mocked_slot_filler.language = language
        config = ProbabilisticIntentParserConfig().to_dict()
        parser_dict = {
            "language_code": "en",
            "intent_classifier": {
                "mocked_dict_key": "mocked_dict_value"
            },
            "slot_fillers": {
                "MakeCoffee": {
                    "mocked_slot_filler_key1": "mocked_slot_filler_value1"
                },
                "MakeTea": {
                    "mocked_slot_filler_key2": "mocked_slot_filler_value2"
                }
            },
            "config": config,
        }

        # When
        parser = ProbabilisticIntentParser.from_dict(parser_dict)

        # Then
        mock_classifier_from_dict.assert_called_once_with(
            {"mocked_dict_key": "mocked_dict_value"})
        calls = [
            call({"mocked_slot_filler_key1": "mocked_slot_filler_value1"}),
            call({"mocked_slot_filler_key2": "mocked_slot_filler_value2"})
        ]
        mock_slot_filler.from_dict.assert_has_calls(calls, any_order=True)

        self.assertEqual(parser.language, language)
        self.assertDictEqual(parser.config.to_dict(), config)
        self.assertIsNotNone(parser.intent_classifier)
        self.assertItemsEqual(parser.slot_fillers.keys(),
                              ["MakeCoffee", "MakeTea"])

    def test_fitting_should_be_reproducible_after_serialization(self):
        # Given
        dataset = BEVERAGE_DATASET
        validated_dataset = validate_and_format_dataset(dataset)

        seed1 = 666
        seed2 = 42
        config = ProbabilisticIntentParserConfig(
            intent_classifier_config=IntentClassifierConfig(random_seed=seed1),
            crf_slot_filler_config=CRFSlotFillerConfig(random_seed=seed2)
        )
        parser = ProbabilisticIntentParser(config)
        parser_dict = parser.to_dict()

        # When
        fitted_parser_1 = ProbabilisticIntentParser.from_dict(
            parser_dict).fit(validated_dataset)

        fitted_parser_2 = ProbabilisticIntentParser.from_dict(
            parser_dict).fit(validated_dataset)

        # Then
        feature_weights_1 = fitted_parser_1.slot_fillers[
            "MakeTea"].crf_model.state_features_
        feature_weights_2 = fitted_parser_2.slot_fillers[
            "MakeTea"].crf_model.state_features_
        self.assertEqual(feature_weights_1, feature_weights_2)
