from __future__ import unicode_literals

from pathlib import Path

from mock import patch

from snips_nlu.constants import RES_INTENT, RES_INTENT_NAME
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier import (
    IntentClassifier, LogRegIntentClassifier)
from snips_nlu.intent_parser import ProbabilisticIntentParser
from snips_nlu.pipeline.configs import (CRFSlotFillerConfig,
                                        LogRegIntentClassifierConfig,
                                        ProcessingUnitConfig,
                                        ProbabilisticIntentParserConfig)
from snips_nlu.pipeline.units_registry import (
    register_processing_unit, reset_processing_units)
from snips_nlu.slot_filler import CRFSlotFiller, SlotFiller
from snips_nlu.tests.utils import BEVERAGE_DATASET, FixtureTest
from snips_nlu.utils import NotTrained, json_string


class TestProbabilisticIntentParser(FixtureTest):
    def setUp(self):
        super(TestProbabilisticIntentParser, self).setUp()
        reset_processing_units()

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

    def test_should_not_parse_when_not_fitted(self):
        # Given
        parser = ProbabilisticIntentParser()

        # When / Then
        self.assertFalse(parser.fitted)
        with self.assertRaises(NotTrained):
            parser.parse("foobar")

    def test_should_be_serializable_before_fitting(self):
        # Given
        parser = ProbabilisticIntentParser()

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_parser_dict = {
            "config": {
                "unit_name": "probabilistic_intent_parser",
                "slot_filler_config": CRFSlotFillerConfig().to_dict(),
                "intent_classifier_config":
                    LogRegIntentClassifierConfig().to_dict()
            },
            "slot_fillers": []
        }
        metadata = {"unit_name": "probabilistic_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json",
                               expected_parser_dict)

    def test_should_be_deserializable_before_fitting(self):
        # When
        config = ProbabilisticIntentParserConfig().to_dict()
        parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "config": config,
            "intent_classifier": None,
            "slot_fillers": dict(),
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "probabilistic_intent_parser"}
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json",
                              parser_dict)

        # When
        parser = ProbabilisticIntentParser.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(parser.config.to_dict(), config)
        self.assertIsNone(parser.intent_classifier)
        self.assertDictEqual(dict(), parser.slot_fillers)

    def test_should_be_serializable(self):
        # Given
        register_processing_unit(TestIntentClassifier)
        register_processing_unit(TestSlotFiller)

        parser_config = ProbabilisticIntentParserConfig(
            intent_classifier_config=TestIntentClassifierConfig(),
            slot_filler_config=TestSlotFillerConfig()
        )
        parser = ProbabilisticIntentParser(parser_config)
        parser.fit(validate_and_format_dataset(BEVERAGE_DATASET))

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_parser_config = {
            "unit_name": "probabilistic_intent_parser",
            "slot_filler_config": {"unit_name": "test_slot_filler"},
            "intent_classifier_config": {"unit_name": "test_intent_classifier"}
        }
        expected_parser_dict = {
            "config": expected_parser_config,
            "slot_fillers": [
                {
                    "intent": "MakeCoffee",
                    "slot_filler_name": "slot_filler_0"
                },
                {
                    "intent": "MakeTea",
                    "slot_filler_name": "slot_filler_1"
                }
            ]
        }
        metadata = {"unit_name": "probabilistic_intent_parser"}
        metadata_slot_filler = {"unit_name": "test_slot_filler"}
        metadata_intent_classifier = {"unit_name": "test_intent_classifier"}

        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json",
                               expected_parser_dict)
        self.assertJsonContent(
            self.tmp_file_path / "intent_classifier" / "metadata.json",
            metadata_intent_classifier)
        self.assertJsonContent(
            self.tmp_file_path / "slot_filler_0" / "metadata.json",
            metadata_slot_filler)
        self.assertJsonContent(
            self.tmp_file_path / "slot_filler_1" / "metadata.json",
            metadata_slot_filler)

    def test_should_be_deserializable(self):
        # When
        register_processing_unit(TestIntentClassifier)
        register_processing_unit(TestSlotFiller)

        config = ProbabilisticIntentParserConfig(
            intent_classifier_config=TestIntentClassifierConfig(),
            slot_filler_config=TestSlotFillerConfig()
        )
        parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "slot_fillers": [
                {
                    "intent": "MakeCoffee",
                    "slot_filler_name": "slot_filler_MakeCoffee"
                },
                {
                    "intent": "MakeTea",
                    "slot_filler_name": "slot_filler_MakeTea"
                }
            ],
            "config": config.to_dict(),
        }
        self.tmp_file_path.mkdir()
        (self.tmp_file_path / "intent_classifier").mkdir()
        (self.tmp_file_path / "slot_filler_MakeCoffee").mkdir()
        (self.tmp_file_path / "slot_filler_MakeTea").mkdir()
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json",
                              parser_dict)
        self.writeJsonContent(
            self.tmp_file_path / "intent_classifier" / "metadata.json",
            {"unit_name": "test_intent_classifier"})
        self.writeJsonContent(
            self.tmp_file_path / "slot_filler_MakeCoffee" / "metadata.json",
            {"unit_name": "test_slot_filler"})
        self.writeJsonContent(
            self.tmp_file_path / "slot_filler_MakeTea" / "metadata.json",
            {"unit_name": "test_slot_filler"})

        # When
        parser = ProbabilisticIntentParser.from_path(self.tmp_file_path)

        # Then
        self.assertDictEqual(parser.config.to_dict(), config.to_dict())
        self.assertIsNotNone(parser.intent_classifier)
        self.assertListEqual(sorted(parser.slot_fillers),
                             ["MakeCoffee", "MakeTea"])

    def test_should_be_serializable_into_bytearray(self):
        # Given
        dataset = BEVERAGE_DATASET
        intent_parser = ProbabilisticIntentParser().fit(dataset)
        builtin_entity_parser = intent_parser.builtin_entity_parser
        custom_entity_parser = intent_parser.custom_entity_parser

        # When
        intent_parser_bytes = intent_parser.to_byte_array()
        loaded_intent_parser = ProbabilisticIntentParser.from_byte_array(
            intent_parser_bytes,
            builtin_entity_parser=builtin_entity_parser,
            custom_entity_parser=custom_entity_parser
        )
        result = loaded_intent_parser.parse("make me two cups of tea")

        # Then
        self.assertEqual("MakeTea", result[RES_INTENT][RES_INTENT_NAME])

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
        parser.persist(self.tmp_file_path)

        # When
        fitted_parser_1 = ProbabilisticIntentParser.from_path(
            self.tmp_file_path).fit(validated_dataset)

        fitted_parser_2 = ProbabilisticIntentParser.from_path(
            self.tmp_file_path).fit(validated_dataset)

        # Then
        feature_weights_1 = fitted_parser_1.slot_fillers[
            "MakeTea"].crf_model.state_features_
        feature_weights_2 = fitted_parser_2.slot_fillers[
            "MakeTea"].crf_model.state_features_
        self.assertEqual(feature_weights_1, feature_weights_2)


class TestIntentClassifierConfig(ProcessingUnitConfig):
    unit_name = "test_intent_classifier"

    def to_dict(self):
        return {"unit_name": self.unit_name}

    @classmethod
    def from_dict(cls, obj_dict):
        return TestIntentClassifierConfig()


# pylint: disable=abstract-method
class TestIntentClassifier(IntentClassifier):
    unit_name = "test_intent_classifier"
    config_type = TestIntentClassifierConfig
    _fitted = False

    @property
    def fitted(self):
        return self._fitted

    def get_intent(self, text, intents_filter):
        return None

    def fit(self, dataset):
        self._fitted = True
        return self

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        with (path / "metadata.json").open(mode="w") as f:
            f.write(json_string({"unit_name": self.unit_name}))

    @classmethod
    def from_path(cls, path, **shared):
        config = cls.config_type()
        return cls(config)


class TestSlotFillerConfig(ProcessingUnitConfig):
    unit_name = "test_slot_filler"

    def to_dict(self):
        return {"unit_name": self.unit_name}

    @classmethod
    def from_dict(cls, obj_dict):
        return TestSlotFillerConfig()


# pylint: disable=abstract-method
class TestSlotFiller(SlotFiller):
    unit_name = "test_slot_filler"
    config_type = TestSlotFillerConfig
    _fitted = False

    @property
    def fitted(self):
        return self._fitted

    def get_slots(self, text):
        return []

    def fit(self, dataset, intent):
        self._fitted = True
        return self

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        with (path / "metadata.json").open(mode="w") as f:
            f.write(json_string({"unit_name": self.unit_name}))

    @classmethod
    def from_path(cls, path, **shared):
        config = cls.config_type()
        return cls(config)
