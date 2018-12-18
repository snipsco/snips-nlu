# coding=utf-8
from __future__ import unicode_literals

import io
from builtins import str

from mock import MagicMock, patch
from snips_nlu_ontology import get_all_languages

import snips_nlu
from snips_nlu.constants import (
    END, LANGUAGE, LANGUAGE_EN, RES_ENTITY, RES_INPUT, RES_INTENT,
    RES_INTENT_NAME, RES_MATCH_RANGE, RES_RAW_VALUE, RES_SLOTS, RES_SLOT_NAME,
    RES_VALUE, START)
from snips_nlu.dataset import Dataset, validate_and_format_dataset
from snips_nlu.entity_parser import BuiltinEntityParser, CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser_usage import \
    CustomEntityParserUsage
from snips_nlu.exceptions import (
    IntentNotFoundError, InvalidInputError, NotTrained)
from snips_nlu.intent_parser import IntentParser
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.pipeline.configs import (
    NLUEngineConfig, ProbabilisticIntentParserConfig)
from snips_nlu.result import (
    custom_slot, empty_result, intent_classification_result, parsing_result,
    resolved_slot, unresolved_slot, extraction_result)
from snips_nlu.tests.utils import (
    FixtureTest, MockIntentParser, get_empty_dataset)


class TestSnipsNLUEngine(FixtureTest):
    def setUp(self):
        super(TestSnipsNLUEngine, self).setUp()

    def test_should_parse_top_intents(self):
        # Given
        text = "foo bar ban"
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - foo [slot1:entity1](bak)
  
---
type: intent
name: intent2
utterances:
  - '[slot2:entity2](foo) baz'
  
---
type: intent
name: intent3
utterances:
  - foo bap""")

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        @IntentParser.register("first_intent_parser", True)
        class FirstIntentParser(MockIntentParser):
            def get_intents(self, text):
                return [
                    intent_classification_result("intent1", 0.5),
                    intent_classification_result("intent2", 0.3),
                    intent_classification_result(None, 0.15),
                    intent_classification_result("intent3", 0.05)
                ]

            def get_slots(self, text, intent):
                if intent == "intent1":
                    return []
                if intent == "intent2":
                    return [
                        unresolved_slot((0, 3), "foo", "entity2", "slot2")
                    ]
                return []

        @IntentParser.register("second_intent_parser", True)
        class SecondIntentParser(MockIntentParser):
            def get_intents(self, text):
                return [
                    intent_classification_result("intent2", 0.6),
                    intent_classification_result("intent1", 0.2),
                    intent_classification_result(None, 0.15),
                    intent_classification_result("intent3", 0.05)
                ]

            def get_slots(self, text, intent):
                if intent == "intent1":
                    return [
                        unresolved_slot((0, 3), "foo", "entity1", "slot1")
                    ]
                if intent == "intent2":
                    return [
                        unresolved_slot((8, 11), "ban", "entity2", "slot2")
                    ]
                return []

        config = NLUEngineConfig(
            ["first_intent_parser", "second_intent_parser"])
        nlu_engine = SnipsNLUEngine(config).fit(dataset)

        # When
        results = nlu_engine.parse(text, top_n=3)

        # Then
        expected_results = [
            extraction_result(
                intent_classification_result("intent2", 0.6),
                [custom_slot(
                    unresolved_slot((0, 3), "foo", "entity2", "slot2"))]
            ),
            extraction_result(
                intent_classification_result("intent1", 0.5),
                [custom_slot(
                    unresolved_slot((0, 3), "foo", "entity1", "slot1"))]
            ),
            extraction_result(
                intent_classification_result(None, 0.15),
                []
            ),
        ]
        self.assertListEqual(expected_results, results)

    def test_should_get_intents(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: greeting1
utterances:
- hello

---
type: intent
name: greeting2
utterances:
- how are you""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        input_text = "hello world"

        @IntentParser.register("first_intent_parser", True)
        class FirstIntentParser(MockIntentParser):
            def get_intents(self, text):
                return [
                    intent_classification_result("greeting1", 0.5),
                    intent_classification_result("greeting2", 0.3),
                    intent_classification_result(None, 0.2)
                ]

        @IntentParser.register("second_intent_parser", True)
        class SecondIntentParser(MockIntentParser):
            def get_intents(self, text):
                return [
                    intent_classification_result("greeting2", 0.6),
                    intent_classification_result("greeting1", 0.2),
                    intent_classification_result(None, 0.1)
                ]

        config = NLUEngineConfig(["first_intent_parser",
                                  "second_intent_parser"])
        engine = SnipsNLUEngine(config).fit(dataset)

        # When
        res_intents = engine.get_intents(input_text)

        # Then
        expected_intents = [
            intent_classification_result("greeting2", 0.6),
            intent_classification_result("greeting1", 0.5),
            intent_classification_result(None, 0.2)
        ]
        self.assertListEqual(expected_intents, res_intents)

    def test_should_get_slots(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: greeting1
utterances:
- hello [greeted:name](john)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        input_text = "hello snips"
        greeting_intent = "greeting"
        expected_slots = [unresolved_slot(match_range=(6, 11),
                                          value="snips",
                                          entity="name",
                                          slot_name="greeted")]

        @IntentParser.register("first_intent_parser", True)
        class FirstIntentParser(MockIntentParser):
            pass

        @IntentParser.register("second_intent_parser", True)
        class SecondIntentParser(MockIntentParser):
            def get_slots(self, text, intent):
                if text == input_text and intent == greeting_intent:
                    return expected_slots
                return []

        config = NLUEngineConfig(
            ["first_intent_parser", "second_intent_parser"])
        engine = SnipsNLUEngine(config).fit(dataset)

        # When
        res_slots = engine.get_slots(input_text, greeting_intent)

        # Then
        expected_slots = [custom_slot(s) for s in expected_slots]
        self.assertListEqual(expected_slots, res_slots)

    def test_get_slots_should_raise_with_unknown_intent(self):
        # Given
        slots_dataset_stream = io.StringIO("""
---
type: intent
name: greeting1
utterances:
  - Hello [name1](John)

---
type: intent
name: goodbye
utterances:
  - Goodbye [name](Eric)""")
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        nlu_engine = SnipsNLUEngine().fit(dataset)

        # When / Then
        with self.assertRaises(IntentNotFoundError):
            nlu_engine.get_slots("Hello John", "greeting3")

    def test_should_use_parsers_sequentially(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: greeting1
utterances:
- hello [greeted:name](john)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        input_text = "hello snips"
        intent = intent_classification_result(
            intent_name='greeting1', probability=0.7)
        slots = [unresolved_slot(match_range=(6, 11),
                                 value='snips',
                                 entity='name',
                                 slot_name='greeted')]

        @IntentParser.register("first_intent_parser", True)
        class FirstIntentParser(MockIntentParser):
            pass

        @IntentParser.register("second_intent_parser", True)
        class SecondIntentParser(MockIntentParser):
            def parse(self, text, intents=None, top_n=None):
                if text == input_text:
                    return parsing_result(text, intent, slots)
                return empty_result(text)

        config = NLUEngineConfig(["first_intent_parser",
                                  "second_intent_parser"])
        engine = SnipsNLUEngine(config).fit(dataset)

        # When
        parse = engine.parse(input_text)

        # Then
        expected_slots = [custom_slot(s) for s in slots]
        expected_parse = parsing_result(input_text, intent, expected_slots)
        self.assertDictEqual(expected_parse, parse)

    def test_should_retrain_only_non_trained_subunits(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: greeting1
utterances:
- hello [greeted:name](john)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        @IntentParser.register("test_intent_parser", True)
        class TestIntentParser(MockIntentParser):
            def __init__(self, config=None, **shared):
                super(TestIntentParser, self).__init__(config, **shared)
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

        nlu_engine_config = NLUEngineConfig(["test_intent_parser"])
        nlu_engine = SnipsNLUEngine(nlu_engine_config)

        intent_parser = TestIntentParser()
        intent_parser.sub_unit_1.update(dict(fitted=True, calls=0))
        nlu_engine.intent_parsers.append(intent_parser)

        # When
        nlu_engine.fit(dataset, force_retrain=False)

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

    def test_should_not_parse_slots_when_not_fitted(self):
        # Given
        engine = SnipsNLUEngine()

        # When / Then
        self.assertFalse(engine.fitted)
        with self.assertRaises(NotTrained):
            engine.parse("foobar")

    def test_should_be_serializable_into_dir(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        @IntentParser.register("test_intent_parser1", True)
        class TestIntentParser1(MockIntentParser):
            pass

        @IntentParser.register("test_intent_parser2", True)
        class TestIntentParser2(MockIntentParser):
            pass

        parser1_config = {
            "unit_name": "test_intent_parser1",
            "my_param1": "foo"
        }
        parser2_config = {
            "unit_name": "test_intent_parser2",
            "my_param2": "bar"
        }
        parsers_configs = [parser1_config, parser2_config]
        config = NLUEngineConfig(parsers_configs)
        engine = SnipsNLUEngine(config).fit(dataset)

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
            "config": {
                "unit_name": "nlu_engine",
                "intent_parsers_configs": [
                    {
                        "unit_name": "test_intent_parser1",
                        "my_param1": "foo"
                    },
                    {
                        "unit_name": "test_intent_parser2",
                        "my_param2": "bar"
                    }
                ]
            },
            "intent_parsers": [
                "test_intent_parser1",
                "test_intent_parser2"
            ],
            "builtin_entity_parser": "builtin_entity_parser",
            "custom_entity_parser": "custom_entity_parser",
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
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        @IntentParser.register("my_intent_parser", True)
        class MyIntentParser(MockIntentParser):
            pass

        parsers_configs = ["my_intent_parser", "my_intent_parser"]
        config = NLUEngineConfig(parsers_configs)
        engine = SnipsNLUEngine(config).fit(dataset)

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
            "config": {
                "unit_name": "nlu_engine",
                "intent_parsers_configs": [
                    {
                        "unit_name": "my_intent_parser"
                    },
                    {
                        "unit_name": "my_intent_parser"
                    }
                ]
            },
            "intent_parsers": [
                "my_intent_parser",
                "my_intent_parser_2"
            ],
            "builtin_entity_parser": "builtin_entity_parser",
            "custom_entity_parser": "custom_entity_parser",
            "model_version": snips_nlu.__model_version__,
            "training_package_version": snips_nlu.__version__
        }
        self.assertJsonContent(self.tmp_file_path / "nlu_engine.json",
                               expected_engine_dict)
        self.assertJsonContent(
            self.tmp_file_path / "my_intent_parser" / "metadata.json",
            {"unit_name": "my_intent_parser"})
        self.assertJsonContent(
            self.tmp_file_path / "my_intent_parser_2" / "metadata.json",
            {"unit_name": "my_intent_parser"})

    @patch("snips_nlu.nlu_engine.nlu_engine.CustomEntityParser")
    @patch("snips_nlu.nlu_engine.nlu_engine.BuiltinEntityParser")
    def test_should_be_deserializable_from_dir(
            self, mocked_builtin_entity_parser, mocked_custom_entity_parser):
        # Given
        mocked_builtin_entity_parser.from_path = MagicMock()
        mocked_custom_entity_parser.from_path = MagicMock()

        @IntentParser.register("test_intent_parser1", True)
        class TestIntentParser1(MockIntentParser):
            pass

        @IntentParser.register("test_intent_parser2", True)
        class TestIntentParser2(MockIntentParser):
            pass

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
        engine_dict = {
            "unit_name": "nlu_engine",
            "dataset_metadata": dataset_metadata,
            "config": {
                "unit_name": "nlu_engine",
                "intent_parsers_configs": [
                    {
                        "unit_name": "test_intent_parser1"
                    },
                    {
                        "unit_name": "test_intent_parser2"
                    }
                ]
            },
            "intent_parsers": [
                "test_intent_parser1",
                "test_intent_parser2",
            ],
            "builtin_entity_parser": "builtin_entity_parser",
            "custom_entity_parser": "custom_entity_parser",
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
        expected_engine_config = {
            "unit_name": "nlu_engine",
            "intent_parsers_configs": [
                {
                    "unit_name": "test_intent_parser1"
                },
                {
                    "unit_name": "test_intent_parser2"
                }
            ]
        }
        # pylint:disable=protected-access
        self.assertDictEqual(dataset_metadata, engine._dataset_metadata)
        # pylint:enable=protected-access
        self.assertDictEqual(expected_engine_config, engine.config.to_dict())
        self.assertIsInstance(engine.intent_parsers[0], TestIntentParser1)
        self.assertIsInstance(engine.intent_parsers[1], TestIntentParser2)
        mocked_custom_entity_parser.from_path.assert_called_once_with(
            self.tmp_file_path / "custom_entity_parser")
        mocked_builtin_entity_parser.from_path.assert_called_once_with(
            self.tmp_file_path / "builtin_entity_parser")

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
            "builtin_entity_parser": None,
            "custom_entity_parser": None,
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
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups
- i want [number_of_cups] cups of [beverage_temperature](boiling hot) tea pls
- can you prepare [number_of_cups] cup of [beverage_temperature](cold) tea ?

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee
- can you prepare [number_of_cups] cup of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
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

    def test_should_be_serializable_into_bytearray_when_empty(self):
        # Given
        engine = SnipsNLUEngine()
        engine_bytes = engine.to_byte_array()

        # When
        engine = SnipsNLUEngine.from_byte_array(engine_bytes)

        # Then
        self.assertFalse(engine.fitted)

    def test_should_be_serializable_into_bytearray(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        engine = SnipsNLUEngine().fit(dataset)

        # When
        engine_bytes = engine.to_byte_array()
        builtin_entity_parser = BuiltinEntityParser.build(dataset=dataset)
        custom_entity_parser = CustomEntityParser.build(
            dataset, parser_usage=CustomEntityParserUsage.WITHOUT_STEMS)
        loaded_engine = SnipsNLUEngine.from_byte_array(
            engine_bytes, builtin_entity_parser=builtin_entity_parser,
            custom_entity_parser=custom_entity_parser)
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
                    ],
                    "matching_strictness": 1.0
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
                    ],
                    "matching_strictness": 1.0
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
                    ],
                    "matching_strictness": 1.0
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
                    "automatically_extensible": False,
                    "matching_strictness": 1.0
                }
            },
            "language": "en",
        }

        mocked_proba_parser_intent = intent_classification_result(
            "intent1", 1.0)

        # pylint: disable=unused-argument
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
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups
- i want [number_of_cups] cups of [beverage_temperature](boiling hot) tea pls
- can you prepare [number_of_cups] cup of [beverage_temperature](cold) tea ?

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee
- can you prepare [number_of_cups] cup of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        text = "please brew me a cup of coffee"
        for language in get_all_languages():
            dataset[LANGUAGE] = language
            engine = SnipsNLUEngine()

            # When / Then
            msg = "Could not fit engine in '%s'" % language
            with self.fail_if_exception(msg):
                engine = engine.fit(dataset)

            msg = "Could not parse in '%s'" % language
            with self.fail_if_exception(msg):
                res = engine.parse(text)
            self.assertEqual("MakeCoffee", res[RES_INTENT][RES_INTENT_NAME])

    def test_nlu_engine_should_raise_error_with_bytes_input(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        bytes_input = b"brew me an espresso"
        engine = SnipsNLUEngine().fit(dataset)

        # When / Then
        with self.assertRaises(InvalidInputError) as cm:
            engine.parse(bytes_input)
        message = str(cm.exception.args[0])
        self.assertTrue("Expected unicode but received" in message)

    def test_should_fit_and_parse_empty_intent(self):
        # Given
        dataset = {
            "intents": {
                "dummy_intent": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": " "
                                }
                            ]
                        }
                    ]
                }
            },
            "language": "en",
            "entities": dict()
        }

        engine = SnipsNLUEngine()

        # When / Then
        engine.fit(dataset)
        engine.parse("ya", intents=["dummy_intent"])
