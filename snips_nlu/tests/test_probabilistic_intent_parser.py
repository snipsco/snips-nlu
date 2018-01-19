from __future__ import unicode_literals

import unittest

from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier.intent_classifier import IntentClassifier
from snips_nlu.intent_parser.intent_parser import NotTrained
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.pipeline.configs.config import ProcessingUnitConfig
from snips_nlu.pipeline.configs.intent_classifier import IntentClassifierConfig
from snips_nlu.pipeline.configs.intent_parser import \
    ProbabilisticIntentParserConfig
from snips_nlu.pipeline.configs.slot_filler import CRFSlotFillerConfig
from snips_nlu.pipeline.units_registry import register_processing_unit
from snips_nlu.slot_filler.slot_filler import SlotFiller
from snips_nlu.tests.utils import BEVERAGE_DATASET


class TestProbabilisticIntentParser(unittest.TestCase):
    def test_should_not_allow_to_fit_with_missing_intents(self):
        # Given
        intent = "MakeTea"
        parser = ProbabilisticIntentParser()
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)

        # When / Then
        with self.assertRaises(NotTrained):
            parser.fit(dataset, {intent})

    def test_should_be_serializable_before_fitting(self):
        # Given
        parser_config = ProbabilisticIntentParserConfig()
        parser = ProbabilisticIntentParser(parser_config)

        # When
        actual_parser_dict = parser.to_dict()

        # Then
        expected_parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "config": {
                "unit_name": "probabilistic_intent_parser",
                "slot_filler_config": CRFSlotFillerConfig().to_dict(),
                "intent_classifier_config": IntentClassifierConfig().to_dict()
            },
            "intent_classifier": None,
            "slot_fillers": None,
        }
        self.assertDictEqual(actual_parser_dict, expected_parser_dict)

    def test_should_be_deserializable_before_fitting(self):
        # When
        config = ProbabilisticIntentParserConfig().to_dict()
        parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "config": config,
            "intent_classifier": None,
            "slot_fillers": None,
        }

        # When
        parser = ProbabilisticIntentParser.from_dict(parser_dict)

        # Then
        self.assertEqual(parser.config.to_dict(), config)
        self.assertIsNone(parser.intent_classifier)
        self.assertIsNone(parser.slot_fillers)

    def test_should_be_serializable(self):
        # Given
        class TestIntentClassifierConfig(ProcessingUnitConfig):
            unit_name = "test_intent_classifier"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentClassifierConfig()

        class TestIntentClassifier(IntentClassifier):
            unit_name = "test_intent_classifier"
            config_type = TestIntentClassifierConfig

            def get_intent(self, text, intents_filter):
                return None

            def fit(self, dataset):
                return self

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                config = cls.config_type()
                return TestIntentClassifier(config)

        class TestSlotFillerConfig(ProcessingUnitConfig):
            unit_name = "test_slot_filler"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestSlotFillerConfig()

        class TestSlotFiller(SlotFiller):
            unit_name = "test_slot_filler"
            config_type = TestSlotFillerConfig

            def get_slots(self, text):
                return []

            def fit(self, dataset, intent):
                return self

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                config = cls.config_type()
                return TestSlotFiller(config)

        register_processing_unit(TestIntentClassifier)
        register_processing_unit(TestSlotFiller)

        parser_config = ProbabilisticIntentParserConfig(
            intent_classifier_config=TestIntentClassifierConfig(),
            slot_filler_config=TestSlotFillerConfig()
        )
        parser = ProbabilisticIntentParser(parser_config)
        parser.fit(validate_and_format_dataset(BEVERAGE_DATASET))

        # When
        actual_parser_dict = parser.to_dict()

        # Then
        expected_parser_config = {
            "unit_name": "probabilistic_intent_parser",
            "slot_filler_config": {"unit_name": "test_slot_filler"},
            "intent_classifier_config": {"unit_name": "test_intent_classifier"}
        }
        expected_parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "config": expected_parser_config,
            "intent_classifier": {"unit_name": "test_intent_classifier"},
            "slot_fillers": {
                "MakeCoffee": {"unit_name": "test_slot_filler"},
                "MakeTea": {"unit_name": "test_slot_filler"},
            },
        }
        self.assertDictEqual(actual_parser_dict, expected_parser_dict)

    def test_should_be_deserializable(self):
        # When
        class TestIntentClassifierConfig(ProcessingUnitConfig):
            unit_name = "test_intent_classifier"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentClassifierConfig()

        class TestIntentClassifier(IntentClassifier):
            unit_name = "test_intent_classifier"
            config_type = TestIntentClassifierConfig

            def get_intent(self, text, intents_filter):
                return None

            def fit(self, dataset):
                return self

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                conf = cls.config_type()
                return TestIntentClassifier(conf)

        class TestSlotFillerConfig(ProcessingUnitConfig):
            unit_name = "test_slot_filler"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestSlotFillerConfig()

        class TestSlotFiller(SlotFiller):
            unit_name = "test_slot_filler"
            config_type = TestSlotFillerConfig

            def get_slots(self, text):
                return []

            def fit(self, dataset, intent):
                return self

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                conf = cls.config_type()
                return TestSlotFiller(conf)

        register_processing_unit(TestIntentClassifier)
        register_processing_unit(TestSlotFiller)

        config = ProbabilisticIntentParserConfig(
            intent_classifier_config=TestIntentClassifierConfig(),
            slot_filler_config=TestSlotFillerConfig()
        )
        parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "intent_classifier": {"unit_name": "test_intent_classifier"},
            "slot_fillers": {
                "MakeCoffee": {"unit_name": "test_slot_filler"},
                "MakeTea": {"unit_name": "test_slot_filler"}
            },
            "config": config.to_dict(),
        }

        # When
        parser = ProbabilisticIntentParser.from_dict(parser_dict)

        # Then
        self.assertDictEqual(parser.config.to_dict(), config.to_dict())
        self.assertIsNotNone(parser.intent_classifier)
        self.assertListEqual(sorted(list(parser.slot_fillers.keys())),
                             ["MakeCoffee", "MakeTea"])

    def test_fitting_should_be_reproducible_after_serialization(self):
        # Given
        dataset = BEVERAGE_DATASET
        validated_dataset = validate_and_format_dataset(dataset)

        seed1 = 666
        seed2 = 42
        config = ProbabilisticIntentParserConfig(
            intent_classifier_config=IntentClassifierConfig(random_seed=seed1),
            slot_filler_config=CRFSlotFillerConfig(random_seed=seed2)
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

    def test_get_fitted_slot_filler_should_return_same_slot_filler_as_fit(
            self):
        # Given
        intent = "MakeCoffee"
        slot_filler_config = CRFSlotFillerConfig(random_seed=42)
        config = ProbabilisticIntentParserConfig(
            slot_filler_config=slot_filler_config)
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        fitted_parser = ProbabilisticIntentParser(config).fit(dataset)

        # When
        parser = ProbabilisticIntentParser(config)
        slot_filler = parser.get_fitted_slot_filler(dataset, intent)

        # Then
        expected_slot_filler = fitted_parser.slot_fillers[intent]
        self.assertEqual(slot_filler.crf_model.state_features_,
                         expected_slot_filler.crf_model.state_features_)
        self.assertEqual(slot_filler.crf_model.transition_features_,
                         expected_slot_filler.crf_model.transition_features_)
