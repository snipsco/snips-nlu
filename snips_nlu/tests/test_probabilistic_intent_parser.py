from __future__ import unicode_literals

from mock import patch

from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier import LogRegIntentClassifier
from snips_nlu.intent_parser import ProbabilisticIntentParser
from snips_nlu.pipeline.configs import (
    CRFSlotFillerConfig, LogRegIntentClassifierConfig,
    ProbabilisticIntentParserConfig, ProcessingUnitConfig)
from snips_nlu.pipeline.units_registry import register_processing_unit
from snips_nlu.slot_filler import CRFSlotFiller, SlotFiller
from snips_nlu.tests.utils import BEVERAGE_DATASET, SnipsTest


class TestProbabilisticIntentParser(SnipsTest):
    def test_should_retrain_intent_classifier_when_force_retrain(self):
        # Given
        parser = ProbabilisticIntentParser()
        intent_classifier = LogRegIntentClassifier()
        intent_classifier.fit(BEVERAGE_DATASET)
        parser.intent_classifier = intent_classifier

        # When / Then
        with patch("snips_nlu.intent_classifier.log_reg_classifier"
                   ".LogRegIntentClassifier.fit") as mock_fit:
            parser.fit(BEVERAGE_DATASET, force_retrain=True)
            mock_fit.assert_called_once()

    def test_should_not_retrain_intent_classifier_when_no_force_retrain(self):
        # Given
        parser = ProbabilisticIntentParser()
        intent_classifier = LogRegIntentClassifier()
        intent_classifier.fit(BEVERAGE_DATASET)
        parser.intent_classifier = intent_classifier

        # When / Then
        with patch("snips_nlu.intent_classifier.log_reg_classifier"
                   ".LogRegIntentClassifier.fit") as mock_fit:
            parser.fit(BEVERAGE_DATASET, force_retrain=False)
            mock_fit.assert_not_called()

    def test_should_retrain_slot_filler_when_force_retrain(self):
        # Given
        parser = ProbabilisticIntentParser()
        slot_filler = CRFSlotFiller()
        slot_filler.fit(BEVERAGE_DATASET, "MakeCoffee")
        parser.slot_fillers["MakeCoffee"] = slot_filler

        # When / Then
        with patch("snips_nlu.slot_filler.crf_slot_filler.CRFSlotFiller.fit") \
                as mock_fit:
            parser.fit(BEVERAGE_DATASET, force_retrain=True)
            self.assertEqual(2, mock_fit.call_count)

    def test_should_not_retrain_slot_filler_when_no_force_retrain(self):
        # Given
        parser = ProbabilisticIntentParser()
        slot_filler = CRFSlotFiller()
        slot_filler.fit(BEVERAGE_DATASET, "MakeCoffee")
        parser.slot_fillers["MakeCoffee"] = slot_filler

        # When / Then
        with patch("snips_nlu.slot_filler.crf_slot_filler.CRFSlotFiller.fit") \
                as mock_fit:
            parser.fit(BEVERAGE_DATASET, force_retrain=False)
            self.assertEqual(1, mock_fit.call_count)

    def test_should_be_serializable_before_fitting(self):
        # Given
        parser = ProbabilisticIntentParser()

        # When
        actual_parser_dict = parser.to_dict()

        # Then
        expected_parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "config": {
                "unit_name": "probabilistic_intent_parser",
                "slot_filler_config": CRFSlotFillerConfig().to_dict(),
                "intent_classifier_config":
                    LogRegIntentClassifierConfig().to_dict()
            },
            "intent_classifier": None,
            "slot_fillers": dict(),
        }
        self.assertDictEqual(actual_parser_dict, expected_parser_dict)

    def test_should_be_deserializable_before_fitting(self):
        # When
        config = ProbabilisticIntentParserConfig().to_dict()
        parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "config": config,
            "intent_classifier": None,
            "slot_fillers": dict(),
        }

        # When
        parser = ProbabilisticIntentParser.from_dict(parser_dict)

        # Then
        self.assertEqual(parser.config.to_dict(), config)
        self.assertIsNone(parser.intent_classifier)
        self.assertDictEqual(dict(), parser.slot_fillers)

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
        self.assertListEqual(sorted(parser.slot_fillers),
                             ["MakeCoffee", "MakeTea"])

    def test_fitting_should_be_reproducible_after_serialization(self):
        # Given
        dataset = BEVERAGE_DATASET
        validated_dataset = validate_and_format_dataset(dataset)

        seed1 = 666
        seed2 = 42
        config = ProbabilisticIntentParserConfig(
            intent_classifier_config=LogRegIntentClassifierConfig(
                random_seed=seed1),
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
