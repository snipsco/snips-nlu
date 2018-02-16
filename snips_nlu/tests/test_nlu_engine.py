# coding=utf-8
from __future__ import unicode_literals

import json
from copy import deepcopy

from builtins import str
from mock import patch

import snips_nlu
import snips_nlu.version
from snips_nlu.constants import (
    LANGUAGE, RES_INTENT, RES_INTENT_NAME, RES_INPUT, RES_SLOTS,
    RES_MATCH_RANGE, RES_RAW_VALUE, RES_VALUE, RES_ENTITY, RES_SLOT_NAME)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser import IntentParser
from snips_nlu.languages import Language
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.pipeline.configs import ProcessingUnitConfig, NLUEngineConfig
from snips_nlu.pipeline.units_registry import (register_processing_unit,
                                               reset_processing_units)
from snips_nlu.result import (
    parsing_result, unresolved_slot, intent_classification_result,
    empty_result,
    custom_slot, resolved_slot)
from snips_nlu.tests.utils import (
    SAMPLE_DATASET, get_empty_dataset, BEVERAGE_DATASET, SnipsTest)


class TestSnipsNLUEngine(SnipsTest):
    def setUp(self):
        reset_processing_units()

    def test_should_use_parsers_sequentially(self):
        # Given
        input_text = "hello world"
        intent = intent_classification_result(
            intent_name='dummy_intent_1', probability=0.7)
        slots = [unresolved_slot(match_range=(6, 11),
                                 value='world',
                                 entity='mocked_entity',
                                 slot_name='mocked_slot_name')]

        class TestIntentParser1Config(ProcessingUnitConfig):
            unit_name = "test_intent_parser1"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentParser1Config()

        class TestIntentParser1(IntentParser):
            unit_name = "test_intent_parser1"
            config_type = TestIntentParser1Config

            def fit(self, dataset):
                return self

            def parse(self, text, intents):
                return empty_result(text)

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                conf = cls.config_type()
                return TestIntentParser1(conf)

        class TestIntentParser2Config(ProcessingUnitConfig):
            unit_name = "test_intent_parser2"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentParser2Config()

        class TestIntentParser2(IntentParser):
            unit_name = "test_intent_parser2"
            config_type = TestIntentParser2Config

            def fit(self, dataset):
                return self

            def parse(self, text, intents):
                if text == input_text:
                    return parsing_result(text, intent, slots)
                return empty_result(text)

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                conf = cls.config_type()
                return TestIntentParser2(conf)

        register_processing_unit(TestIntentParser1)
        register_processing_unit(TestIntentParser2)

        mocked_dataset_metadata = {
            "language_code": "en",
            "entities": {
                "mocked_entity": {
                    "automatically_extensible": True,
                    "utterances": dict()
                }
            },
            "slot_name_mappings": {
                "dummy_intent_1": {
                    "mocked_slot_name": "mocked_entity"
                }
            }
        }

        config = NLUEngineConfig([TestIntentParser1Config(),
                                  TestIntentParser2Config()])
        engine = SnipsNLUEngine(config).fit(SAMPLE_DATASET)
        # pylint:disable=protected-access
        engine._dataset_metadata = mocked_dataset_metadata
        # pylint:enable=protected-access

        # When
        parse = engine.parse(input_text)

        # Then
        expected_slots = [custom_slot(s) for s in slots]
        expected_parse = parsing_result(input_text, intent, expected_slots)
        self.assertDictEqual(expected_parse, parse)

    def test_should_handle_empty_dataset(self):
        # Given
        dataset = validate_and_format_dataset(get_empty_dataset(Language.EN))
        engine = SnipsNLUEngine().fit(dataset)

        # When
        result = engine.parse("hello world")

        # Then
        self.assertEqual(empty_result("hello world"), result)

    def test_should_be_serializable(self):
        # Given
        class TestIntentParser1Config(ProcessingUnitConfig):
            unit_name = "test_intent_parser1"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentParser1Config()

        class TestIntentParser1(IntentParser):
            unit_name = "test_intent_parser1"
            config_type = TestIntentParser1Config

            def fit(self, dataset):
                return self

            def parse(self, text, intents):
                return empty_result(text)

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                conf = cls.config_type()
                return TestIntentParser1(conf)

        class TestIntentParser2Config(ProcessingUnitConfig):
            unit_name = "test_intent_parser2"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentParser2Config()

        class TestIntentParser2(IntentParser):
            unit_name = "test_intent_parser2"
            config_type = TestIntentParser2Config

            def fit(self, dataset):
                return self

            def parse(self, text, intents):
                return empty_result(text)

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                conf = cls.config_type()
                return TestIntentParser2(conf)

        register_processing_unit(TestIntentParser1)
        register_processing_unit(TestIntentParser2)

        parser1_config = TestIntentParser1Config()
        parser2_config = TestIntentParser2Config()
        parsers_configs = [parser1_config, parser2_config]
        config = NLUEngineConfig(parsers_configs)
        engine = SnipsNLUEngine(config).fit(BEVERAGE_DATASET)

        # When
        actual_engine_dict = engine.to_dict()

        # Then
        parser1_config = TestIntentParser1Config()
        parser2_config = TestIntentParser2Config()
        parsers_configs = [parser1_config, parser2_config]
        expected_engine_config = NLUEngineConfig(parsers_configs)
        expected_engine_dict = {
            "unit_name": "nlu_engine",
            "dataset_metadata": {
                "language_code": "en",
                "entities": {
                    "Temperature": {
                        "automatically_extensible": True,
                        "utterances": {
                            "boiling": "hot",
                            "cold": "cold",
                            "hot": "hot",
                            "iced": "cold"
                        }
                    }
                },
                "slot_name_mappings": {
                    "MakeCoffee": {
                        "number_of_cups": "snips/number"
                    },
                    "MakeTea": {
                        "beverage_temperature": "Temperature",
                        "number_of_cups": "snips/number"
                    }
                },
            },
            "config": expected_engine_config.to_dict(),
            "intent_parsers": [
                {"unit_name": "test_intent_parser1"},
                {"unit_name": "test_intent_parser2"}
            ],
            "model_version": snips_nlu.version.__model_version__,
            "training_package_version": snips_nlu.__version__
        }

        self.assertDictEqual(actual_engine_dict, expected_engine_dict)

    def test_should_be_deserializable(self):
        # When
        class TestIntentParser1Config(ProcessingUnitConfig):
            unit_name = "test_intent_parser1"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentParser1Config()

        class TestIntentParser1(IntentParser):
            unit_name = "test_intent_parser1"
            config_type = TestIntentParser1Config

            def fit(self, dataset):
                return self

            def parse(self, text, intents):
                return empty_result(text)

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                config = cls.config_type()
                return TestIntentParser1(config)

        class TestIntentParser2Config(ProcessingUnitConfig):
            unit_name = "test_intent_parser2"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentParser2Config()

        class TestIntentParser2(IntentParser):
            unit_name = "test_intent_parser2"
            config_type = TestIntentParser2Config

            def fit(self, dataset):
                return self

            def parse(self, text, intents):
                return empty_result(text)

            def to_dict(self):
                return {
                    "unit_name": self.unit_name,
                }

            @classmethod
            def from_dict(cls, unit_dict):
                config = cls.config_type()
                return TestIntentParser2(config)

        register_processing_unit(TestIntentParser1)
        register_processing_unit(TestIntentParser2)

        dataset_metadata = {
            "language_code": "en",
            "entities": {
                "Temperature": {
                    "automatically_extensible": True,
                    "utterances": {
                        "boiling": "hot",
                        "cold": "cold",
                        "hot": "hot",
                        "iced": "cold"
                    }
                }
            },
            "slot_name_mappings": {
                "MakeCoffee": {
                    "number_of_cups": "snips/number"
                },
                "MakeTea": {
                    "beverage_temperature": "Temperature",
                    "number_of_cups": "snips/number"
                }
            },
        }
        parser1_config = TestIntentParser1Config()
        parser2_config = TestIntentParser2Config()
        engine_config = NLUEngineConfig([parser1_config, parser2_config])
        engine_dict = {
            "unit_name": "nlu_engine",
            "dataset_metadata": dataset_metadata,
            "config": engine_config.to_dict(),
            "intent_parsers": [
                {"unit_name": "test_intent_parser1"},
                {"unit_name": "test_intent_parser2"},
            ],
            "model_version": snips_nlu.version.__model_version__,
            "training_package_version": snips_nlu.__version__
        }
        engine = SnipsNLUEngine.from_dict(engine_dict)

        # Then
        parser1_config = TestIntentParser1Config()
        parser2_config = TestIntentParser2Config()
        expected_engine_config = NLUEngineConfig(
            [parser1_config, parser2_config]).to_dict()
        # pylint:disable=protected-access
        self.assertDictEqual(engine._dataset_metadata, dataset_metadata)
        # pylint:enable=protected-access
        self.assertDictEqual(engine.config.to_dict(), expected_engine_config)

    def test_should_parse_after_deserialization(self):
        # Given
        dataset = BEVERAGE_DATASET
        engine = SnipsNLUEngine().fit(dataset)
        input_ = "Give me 3 cups of hot tea please"

        # When
        engine_dict = engine.to_dict()
        deserialized_engine = SnipsNLUEngine.from_dict(engine_dict)
        result = deserialized_engine.parse(input_)

        # Then
        msg = "SnipsNLUEngine dict should be json serializable to utf-8"
        with self.fail_if_exception(msg):
            json.dumps(engine_dict).encode("utf-8")
        expected_slots = [
            resolved_slot((8, 9), '3', {'type': 'value', 'value': 3},
                          'snips/number', 'number_of_cups'),
            custom_slot(unresolved_slot((18, 21), 'hot', 'Temperature',
                                        'beverage_temperature'))
        ]
        self.assertEqual(result[RES_INPUT], input_)
        self.assertEqual(result[RES_INTENT][RES_INTENT_NAME], 'MakeTea')
        self.assertListEqual(result[RES_SLOTS], expected_slots)

    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.parse")
    @patch("snips_nlu.intent_parser.deterministic_intent_parser"
           ".DeterministicIntentParser.parse")
    def test_should_handle_keyword_entities(self, mocked_regex_parse,
                                            mocked_crf_parse):
        # Given
        dataset = {
            "snips_nlu_version": "1.1.1",
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy_1",
                                    "entity": "dummy_entity_1",
                                    "slot_name": "dummy_slot_name"
                                },
                                {
                                    "text": " dummy_2",
                                    "entity": "dummy_entity_2",
                                    "slot_name": "other_dummy_slot_name"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "dummy_entity_1": {
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "data": [
                        {
                            "value": "dummy1",
                            "synonyms": [
                                "dummy1",
                                "dummy1_bis"
                            ]
                        },
                        {
                            "value": "dummy2",
                            "synonyms": [
                                "dummy2",
                                "dummy2_bis"
                            ]
                        }
                    ]
                },
                "dummy_entity_2": {
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "data": [
                        {
                            "value": "dummy2",
                            "synonyms": [
                                "dummy2"
                            ]
                        }
                    ]
                }
            },
            "language": "en"
        }

        text = "dummy_3 dummy_4"
        mocked_crf_intent = intent_classification_result("dummy_intent_1", 1.0)
        mocked_crf_slots = [unresolved_slot(match_range=(0, 7),
                                            value="dummy_3",
                                            entity="dummy_entity_1",
                                            slot_name="dummy_slot_name"),
                            unresolved_slot(match_range=(8, 15),
                                            value="dummy_4",
                                            entity="dummy_entity_2",
                                            slot_name="other_dummy_slot_name")]

        mocked_regex_parse.return_value = empty_result(text)
        mocked_crf_parse.return_value = parsing_result(
            text, mocked_crf_intent, mocked_crf_slots)

        engine = SnipsNLUEngine()

        # When
        engine = engine.fit(dataset)
        result = engine.parse(text)

        # Then
        expected_slot = custom_slot(unresolved_slot(
            match_range=(8, 15), value="dummy_4", entity="dummy_entity_2",
            slot_name="other_dummy_slot_name"))
        expected_result = parsing_result(text, intent=mocked_crf_intent,
                                         slots=[expected_slot])
        self.assertEqual(expected_result, result)

    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.parse")
    @patch("snips_nlu.intent_parser.deterministic_intent_parser"
           ".DeterministicIntentParser.parse")
    def test_synonyms_should_point_to_base_value(self, mocked_deter_parse,
                                                 mocked_proba_parse):
        # Given
        dataset = {
            "snips_nlu_version": "1.1.1",
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy_1",
                                    "entity": "dummy_entity_1",
                                    "slot_name": "dummy_slot_name"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "dummy_entity_1": {
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "data": [
                        {
                            "value": "dummy1",
                            "synonyms": [
                                "dummy1",
                                "dummy1_bis"
                            ]
                        }
                    ]
                }
            },
            "language": "en"
        }

        text = "dummy1_bis"
        mocked_proba_parser_intent = intent_classification_result(
            "dummy_intent_1", 1.0)
        mocked_proba_parser_slots = [
            unresolved_slot(match_range=(0, 10), value="dummy1_bis",
                            entity="dummy_entity_1",
                            slot_name="dummy_slot_name")]

        mocked_deter_parse.return_value = empty_result(text)
        mocked_proba_parse.return_value = parsing_result(
            text, mocked_proba_parser_intent, mocked_proba_parser_slots)

        engine = SnipsNLUEngine().fit(dataset)

        # When
        result = engine.parse(text)

        # Then
        expected_slot = {
            RES_MATCH_RANGE: {
                "start": 0,
                "end": 10
            },
            RES_RAW_VALUE: "dummy1_bis",
            RES_VALUE: {
                "kind": "Custom",
                "value": "dummy1"
            },
            RES_ENTITY: "dummy_entity_1",
            RES_SLOT_NAME: "dummy_slot_name"
        }
        expected_result = parsing_result(
            text, intent=mocked_proba_parser_intent, slots=[expected_slot])
        self.assertEqual(expected_result, result)

    def test_engine_should_fit_with_builtins_entities(self):
        # Given
        dataset = validate_and_format_dataset({
            "intents": {
                "dummy": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "10p.m.",
                                    "entity": "snips/datetime",
                                    "slot_name": "startTime"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "snips/datetime": {}
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        })

        # When / Then
        SnipsNLUEngine().fit(dataset)  # This should not raise any error

    def test_nlu_engine_should_train_and_parse_in_all_languages(self):
        # Given
        text = "brew me an expresso"
        for l in Language:
            dataset = deepcopy(BEVERAGE_DATASET)
            dataset[LANGUAGE] = l.iso_code
            engine = SnipsNLUEngine()

            # When / Then
            msg = "Could not fit engine in '%s'" % l.iso_code
            with self.fail_if_exception(msg):
                engine = engine.fit(dataset)

            msg = "Could not parse in '%s'" % l.iso_code
            with self.fail_if_exception(msg):
                engine.parse(text)

    def test_nlu_engine_should_raise_error_with_bytes_input(self):
        # Given
        bytes_input = b"brew me an espresso"
        engine = SnipsNLUEngine().fit(BEVERAGE_DATASET)

        # When / Then
        with self.assertRaises(TypeError) as cm:
            engine.parse(bytes_input)
        message = str(cm.exception.args[0])
        self.assertTrue("Expected unicode but received" in message)
