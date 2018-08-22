# coding=utf-8
from __future__ import unicode_literals

from builtins import str
from copy import deepcopy
from pathlib import Path

from mock import patch
from snips_nlu_ontology import get_all_languages

import snips_nlu
from snips_nlu.constants import (
    END, LANGUAGE, LANGUAGE_EN, RES_ENTITY, RES_INPUT, RES_INTENT,
    RES_INTENT_NAME, RES_MATCH_RANGE, RES_RAW_VALUE, RES_SLOTS, RES_SLOT_NAME,
    RES_VALUE, START)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser import IntentParser
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.pipeline.configs import NLUEngineConfig, \
    ProbabilisticIntentParserConfig, ProcessingUnitConfig
from snips_nlu.pipeline.units_registry import (
    register_processing_unit, reset_processing_units)
from snips_nlu.result import (
    custom_slot, empty_result, intent_classification_result, parsing_result,
    resolved_slot, unresolved_slot)
from snips_nlu.tests.utils import (
    BEVERAGE_DATASET, FixtureTest, SAMPLE_DATASET, get_empty_dataset)
from snips_nlu.utils import json_string


class TestSnipsNLUEngine(FixtureTest):
    def setUp(self):
        super(TestSnipsNLUEngine, self).setUp()
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

        class FirstIntentParserConfig(ProcessingUnitConfig):
            unit_name = "first_intent_parser"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return FirstIntentParserConfig()

            def get_required_resources(self):
                return None

        class FirstIntentParser(IntentParser):
            unit_name = "first_intent_parser"
            config_type = FirstIntentParserConfig

            def fit(self, dataset, force_retrain):
                self._fitted = True
                return self

            @property
            def fitted(self):
                return hasattr(self, '_fitted') and self._fitted

            def parse(self, text, intents):
                return empty_result(text)

            def persist(self, path):
                path = Path(path)
                path.mkdir()
                with (path / "metadata.json").open(mode="w") as f:
                    f.write(json_string({"unit_name": self.unit_name}))

            @classmethod
            def from_path(cls, path):
                cfg = cls.config_type()
                return cls(cfg)

        class SecondIntentParserConfig(ProcessingUnitConfig):
            unit_name = "second_intent_parser"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return SecondIntentParserConfig()

            def get_required_resources(self):
                return None

        class SecondIntentParser(IntentParser):
            unit_name = "second_intent_parser"
            config_type = SecondIntentParserConfig

            def fit(self, dataset, force_retrain):
                self._fitted = True
                return self

            @property
            def fitted(self):
                return hasattr(self, '_fitted') and self._fitted

            def parse(self, text, intents):
                if text == input_text:
                    return parsing_result(text, intent, slots)
                return empty_result(text)

            def persist(self, path):
                path = Path(path)
                path.mkdir()
                with (path / "metadata.json").open(mode="w") as f:
                    f.write(json_string({"unit_name": self.unit_name}))

            @classmethod
            def from_path(cls, path):
                cfg = cls.config_type()
                return cls(cfg)

        register_processing_unit(FirstIntentParser)
        register_processing_unit(SecondIntentParser)

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

        config = NLUEngineConfig([FirstIntentParserConfig(),
                                  SecondIntentParserConfig()])
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

    def test_should_retrain_only_non_trained_subunits(self):
        # Given
        class TestIntentParserConfig(ProcessingUnitConfig):
            unit_name = "test_intent_parser"

            def to_dict(self):
                return {"unit_name": self.unit_name}

            @classmethod
            def from_dict(cls, obj_dict):
                return TestIntentParserConfig()

            def get_required_resources(self):
                return None

        class TestIntentParser(IntentParser):
            unit_name = "test_intent_parser"
            config_type = TestIntentParserConfig

            def __init__(self, config):
                super(TestIntentParser, self).__init__(config)
                self.sub_unit_1 = dict(fitted=False, calls=0)
                self.sub_unit_2 = dict(fitted=False, calls=0)

            def fit(self, dataset, force_retrain):
                if force_retrain:
                    self.sub_unit_1["fitted"] = True
                    self.sub_unit_1["calls"] += 1
                    self.sub_unit_2["fitted"] = True
                    self.sub_unit_2["calls"] += 1
                else:
                    if not self.sub_unit_1["fitted"]:
                        self.sub_unit_1["fitted"] = True
                        self.sub_unit_1["calls"] += 1
                    if not self.sub_unit_2["fitted"]:
                        self.sub_unit_2["fitted"] = True
                        self.sub_unit_2["calls"] += 1

                return self

            @property
            def fitted(self):
                return self.sub_unit_1["fitted"] and \
                       self.sub_unit_2["fitted"]

            def parse(self, text, intents):
                return empty_result(text)

            def persist(self, path):
                path = Path(path)
                path.mkdir()
                with (path / "metadata.json").open(mode="w") as f:
                    f.write(json_string({"unit_name": self.unit_name}))

            @classmethod
            def from_path(cls, path):
                cfg = cls.config_type()
                return cls(cfg)

        register_processing_unit(TestIntentParser)

        intent_parser_config = TestIntentParserConfig()
        nlu_engine_config = NLUEngineConfig([intent_parser_config])
        nlu_engine = SnipsNLUEngine(nlu_engine_config)

        intent_parser = TestIntentParser(intent_parser_config)
        intent_parser.sub_unit_1.update(dict(fitted=True, calls=0))
        nlu_engine.intent_parsers.append(intent_parser)

        # When
        nlu_engine.fit(SAMPLE_DATASET, force_retrain=False)

        # Then
        self.assertDictEqual(dict(fitted=True, calls=0),
                             intent_parser.sub_unit_1)
        self.assertDictEqual(dict(fitted=True, calls=1),
                             intent_parser.sub_unit_2)

    def test_should_handle_empty_dataset(self):
        # Given
        dataset = validate_and_format_dataset(get_empty_dataset(LANGUAGE_EN))
        engine = SnipsNLUEngine().fit(dataset)

        # When
        result = engine.parse("hello world")

        # Then
        self.assertEqual(empty_result("hello world"), result)

    def test_should_be_serializable_into_zip(self):
        # Given
        register_processing_unit(TestIntentParser1)
        register_processing_unit(TestIntentParser2)

        parser1_config = TestIntentParser1Config()
        parser2_config = TestIntentParser2Config()
        parsers_configs = [parser1_config, parser2_config]
        config = NLUEngineConfig(parsers_configs)
        engine = SnipsNLUEngine(config).fit(BEVERAGE_DATASET)

        # When
        engine.persist(self.tmp_file_path)

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
                            "Boiling": "hot",
                            "cold": "cold",
                            "Cold": "cold",
                            "hot": "hot",
                            "Hot": "hot",
                            "iced": "cold",
                            "Iced": "cold"
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
                "test_intent_parser1",
                "test_intent_parser2"
            ],
            "model_version": snips_nlu.__model_version__,
            "training_package_version": snips_nlu.__version__
        }
        self.assertJsonContent(self.tmp_file_path / "nlu_engine.json",
                               expected_engine_dict)
        self.assertJsonContent(
            self.tmp_file_path / "test_intent_parser1" / "metadata.json",
            {"unit_name": "test_intent_parser1"})
        self.assertJsonContent(
            self.tmp_file_path / "test_intent_parser2" / "metadata.json",
            {"unit_name": "test_intent_parser2"})

    def test_should_serialize_duplicated_intent_parsers(self):
        # Given
        register_processing_unit(TestIntentParser1)
        parser1_config = TestIntentParser1Config()
        parser1bis_config = TestIntentParser1Config()

        parsers_configs = [parser1_config, parser1bis_config]
        config = NLUEngineConfig(parsers_configs)
        engine = SnipsNLUEngine(config).fit(BEVERAGE_DATASET)

        # When
        engine.persist(self.tmp_file_path)

        # Then
        expected_engine_dict = {
            "unit_name": "nlu_engine",
            "dataset_metadata": {
                "language_code": "en",
                "entities": {
                    "Temperature": {
                        "automatically_extensible": True,
                        "utterances": {
                            "boiling": "hot",
                            "Boiling": "hot",
                            "cold": "cold",
                            "Cold": "cold",
                            "hot": "hot",
                            "Hot": "hot",
                            "iced": "cold",
                            "Iced": "cold"
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
            "config": config.to_dict(),
            "intent_parsers": [
                "test_intent_parser1",
                "test_intent_parser1_2"
            ],
            "model_version": snips_nlu.__model_version__,
            "training_package_version": snips_nlu.__version__
        }
        self.assertJsonContent(self.tmp_file_path / "nlu_engine.json",
                               expected_engine_dict)
        self.assertJsonContent(
            self.tmp_file_path / "test_intent_parser1" / "metadata.json",
            {"unit_name": "test_intent_parser1"})
        self.assertJsonContent(
            self.tmp_file_path / "test_intent_parser1_2" / "metadata.json",
            {"unit_name": "test_intent_parser1"})

    def test_should_be_deserializable_from_dir(self):
        # Given
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
                "test_intent_parser1",
                "test_intent_parser2",
            ],
            "model_version": snips_nlu.__model_version__,
            "training_package_version": snips_nlu.__version__
        }
        self.tmp_file_path.mkdir()
        parser1_path = self.tmp_file_path / "test_intent_parser1"
        parser1_path.mkdir()
        parser2_path = self.tmp_file_path / "test_intent_parser2"
        parser2_path.mkdir()
        (self.tmp_file_path / "resources").mkdir()
        self.writeJsonContent(self.tmp_file_path / "nlu_engine.json",
                              engine_dict)
        self.writeJsonContent(parser1_path / "metadata.json",
                              {"unit_name": "test_intent_parser1"})
        self.writeJsonContent(parser2_path / "metadata.json",
                              {"unit_name": "test_intent_parser2"})

        # When
        engine = SnipsNLUEngine.from_path(self.tmp_file_path)

        # Then
        parser1_config = TestIntentParser1Config()
        parser2_config = TestIntentParser2Config()
        expected_engine_config = NLUEngineConfig(
            [parser1_config, parser2_config]).to_dict()
        # pylint:disable=protected-access
        self.assertDictEqual(engine._dataset_metadata, dataset_metadata)
        # pylint:enable=protected-access
        self.assertDictEqual(engine.config.to_dict(), expected_engine_config)

    def test_should_be_serializable_into_dir_when_empty(self):
        # Given
        nlu_engine = SnipsNLUEngine()

        # When
        nlu_engine.persist(self.tmp_file_path)

        # Then
        expected_dict = {
            "unit_name": "nlu_engine",
            "dataset_metadata": None,
            "config": None,
            "intent_parsers": [],
            "model_version": snips_nlu.__model_version__,
            "training_package_version": snips_nlu.__version__
        }
        self.assertJsonContent(self.tmp_file_path / "nlu_engine.json",
                               expected_dict)

    def test_should_be_deserializable_from_dir_when_empty(self):
        # Given
        engine = SnipsNLUEngine()
        engine.persist(self.tmp_file_path)

        # When
        engine = SnipsNLUEngine.from_path(self.tmp_file_path)

        # Then
        self.assertFalse(engine.fitted)

    def test_should_parse_after_deserialization_from_dir(self):
        # Given
        dataset = BEVERAGE_DATASET
        engine = SnipsNLUEngine().fit(dataset)
        input_ = "Give me 3 cups of hot tea please"

        # When
        engine.persist(self.tmp_file_path)
        deserialized_engine = SnipsNLUEngine.from_path(self.tmp_file_path)
        result = deserialized_engine.parse(input_)

        # Then
        expected_slots = [
            resolved_slot({START: 8, END: 9}, "3",
                          {"kind": "Number", "value": 3.0},
                          "snips/number", "number_of_cups"),
            custom_slot(
                unresolved_slot({START: 18, END: 21}, "hot", "Temperature",
                                "beverage_temperature"))
        ]
        self.assertEqual(result[RES_INPUT], input_)
        self.assertEqual(result[RES_INTENT][RES_INTENT_NAME], "MakeTea")
        self.assertListEqual(result[RES_SLOTS], expected_slots)

    def test_should_be_serializable_into_bytearray(self):
        # Given
        dataset = BEVERAGE_DATASET
        engine = SnipsNLUEngine().fit(dataset)

        # When
        engine_bytes = engine.to_byte_array()
        loaded_engine = SnipsNLUEngine.from_byte_array(engine_bytes)
        result = loaded_engine.parse("Make me two cups of coffee")

        # Then
        self.assertEqual(result[RES_INTENT][RES_INTENT_NAME], "MakeCoffee")

    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.parse")
    @patch("snips_nlu.intent_parser.deterministic_intent_parser"
           ".DeterministicIntentParser.parse")
    def test_should_handle_keyword_entities(self, mocked_regex_parse,
                                            mocked_crf_parse):
        # Given
        dataset = {
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
    def test_synonyms_should_point_to_base_value(self, mocked_proba_parse):
        # Given
        dataset = {
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

        mocked_proba_parse.return_value = parsing_result(
            text, mocked_proba_parser_intent, mocked_proba_parser_slots)

        config = NLUEngineConfig([ProbabilisticIntentParserConfig()])
        engine = SnipsNLUEngine(config).fit(dataset)

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

    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.parse")
    def test_synonyms_should_not_collide_when_remapped_to_base_value(
            self, mocked_proba_parse):
        # Given
        # Given
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "value",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "a",
                            "synonyms": [
                                "favorïte"
                            ]
                        },
                        {
                            "value": "b",
                            "synonyms": [
                                "favorite"
                            ]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en",
        }

        mocked_proba_parser_intent = intent_classification_result(
            "intent1", 1.0)

        def mock_proba_parse(text, intents):
            slots = [unresolved_slot(match_range=(0, len(text)), value=text,
                                     entity="entity1", slot_name="slot1")]
            return parsing_result(
                text, mocked_proba_parser_intent, slots)

        mocked_proba_parse.side_effect = mock_proba_parse

        config = NLUEngineConfig([ProbabilisticIntentParserConfig()])
        engine = SnipsNLUEngine(config).fit(dataset)

        # When
        result1 = engine.parse("favorite")
        result2 = engine.parse("favorïte")

        # Then
        expected_slot1 = {
            RES_MATCH_RANGE: {
                "start": 0,
                "end": 8
            },
            RES_RAW_VALUE: "favorite",
            RES_VALUE: {
                "kind": "Custom",
                "value": "b"
            },
            RES_ENTITY: "entity1",
            RES_SLOT_NAME: "slot1"
        }
        expected_slot2 = {
            RES_MATCH_RANGE: {
                "start": 0,
                "end": 8
            },
            RES_RAW_VALUE: "favorïte",
            RES_VALUE: {
                "kind": "Custom",
                "value": "a"
            },
            RES_ENTITY: "entity1",
            RES_SLOT_NAME: "slot1"
        }
        expected_result1 = parsing_result(
            "favorite", intent=mocked_proba_parser_intent,
            slots=[expected_slot1])
        expected_result2 = parsing_result(
            "favorïte", intent=mocked_proba_parser_intent,
            slots=[expected_slot2])
        self.assertEqual(expected_result1, result1)
        self.assertEqual(expected_result2, result2)

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
        })

        # When / Then
        SnipsNLUEngine().fit(dataset)  # This should not raise any error

    def test_nlu_engine_should_train_and_parse_in_all_languages(self):
        # Given
        text = "brew me an espresso"
        for language in get_all_languages():
            dataset = deepcopy(BEVERAGE_DATASET)
            dataset[LANGUAGE] = language
            engine = SnipsNLUEngine()

            # When / Then
            msg = "Could not fit engine in '%s'" % language
            with self.fail_if_exception(msg):
                engine = engine.fit(dataset)

            msg = "Could not parse in '%s'" % language
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


class TestIntentParser1Config(ProcessingUnitConfig):
    unit_name = "test_intent_parser1"

    def to_dict(self):
        return {"unit_name": self.unit_name}

    @classmethod
    def from_dict(cls, obj_dict):
        return TestIntentParser1Config()

    def get_required_resources(self):
        return None


class TestIntentParser1(IntentParser):
    unit_name = "test_intent_parser1"
    config_type = TestIntentParser1Config

    def fit(self, dataset, force_retrain):
        self._fitted = True
        return self

    @property
    def fitted(self):
        return hasattr(self, '_fitted') and self._fitted

    def parse(self, text, intents):
        return empty_result(text)

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        with (path / "metadata.json").open(mode="w") as f:
            f.write(json_string({"unit_name": self.unit_name}))

    @classmethod
    def from_path(cls, path):
        cfg = cls.config_type()
        return cls(cfg)


class TestIntentParser2Config(ProcessingUnitConfig):
    unit_name = "test_intent_parser2"

    def to_dict(self):
        return {"unit_name": self.unit_name}

    @classmethod
    def from_dict(cls, obj_dict):
        return TestIntentParser2Config()

    def get_required_resources(self):
        return None


class TestIntentParser2(IntentParser):
    unit_name = "test_intent_parser2"
    config_type = TestIntentParser2Config

    def fit(self, dataset, force_retrain):
        self._fitted = True
        return self

    @property
    def fitted(self):
        return hasattr(self, '_fitted') and self._fitted

    def parse(self, text, intents):
        return empty_result(text)

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        with (path / "metadata.json").open(mode="w") as f:
            f.write(json_string({"unit_name": self.unit_name}))

    @classmethod
    def from_path(cls, path):
        cfg = cls.config_type()
        return cls(cfg)
