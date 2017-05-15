from __future__ import unicode_literals

import io
import json
import os
import shutil
import unittest

from mock import Mock, patch

from snips_nlu.constants import ENGINE_TYPE, CUSTOM_ENGINE, DATA, TEXT
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.nlu_engine import SnipsNLUEngine, enrich_slots
from snips_nlu.result import Result, ParsedSlot, IntentClassificationResult
from utils import SAMPLE_DATASET, empty_dataset, TEST_PATH, BEVERAGE_DATASET


class TestSnipsNLUEngine(unittest.TestCase):
    def setUp(self):
        fixtures_directory = os.path.join(TEST_PATH, "fixtures", "nlu_engine")
        self.expected_engine_directory = os.path.join(fixtures_directory,
                                                      "expected_output")
        self.actual_engine_directory = os.path.join(fixtures_directory,
                                                    "actual_output")

    def tearDown(self):
        if os.path.isdir(self.actual_engine_directory):
            shutil.rmtree(self.actual_engine_directory)

    def test_should_use_parsers_sequentially(self):
        # Given
        language = Language.EN

        input_text = "hello world"

        mocked_parser1 = Mock()
        intent_result1 = None
        intent_entities1 = []
        mocked_parser1.get_intent.return_value = intent_result1
        mocked_parser1.get_entities.return_value = intent_entities1

        mocked_parser2 = Mock()
        intent_result2 = IntentClassificationResult(
            intent_name='mocked_intent2', probability=0.7)
        intent_entities2_empty = []
        intent_entities2 = [
            ParsedSlot(match_range=(3, 5), value='mocked_value',
                       entity='mocked_entity', slot_name='mocked_slot_name')]
        mocked_parser2.get_intent.return_value = intent_result2

        def mock_get_slots(text, intent):
            assert text == input_text
            if intent == intent_result2.intent_name:
                return intent_entities2
            else:
                return intent_entities2_empty

        mocked_parser2.get_slots = Mock(side_effect=mock_get_slots)
        mocked_entities = {"mocked_entity": {"automatically_extensible": True}}
        engine = SnipsNLUEngine(
            language, entities=mocked_entities,
            rule_based_parser=mocked_parser1,
            probabilistic_parser=mocked_parser2,
            slot_name_mapping={'mocked_slot_name': 'mocked_entity'})

        # When
        parse = engine.parse(input_text)

        # Then
        expected_parse = Result(input_text, intent_result2,
                                intent_entities2).as_dict()
        self.assertEqual(parse, expected_parse)

    def test_should_handle_empty_dataset(self):
        # Given
        engine = SnipsNLUEngine(Language.EN).fit(empty_dataset(Language.EN))

        # When
        result = engine.parse("hello world")

        # Then
        self.assertEqual(result, Result("hello world", None, None).as_dict())

    @patch('snips_nlu.nlu_engine.ProbabilisticIntentParser.save')
    @patch('snips_nlu.nlu_engine.RegexIntentParser.save')
    def test_should_be_saveable(self, mock_rule_based_parser_save,
                                mock_probabilistic_parser_save):
        # Given
        language = Language.EN
        engine = SnipsNLUEngine(
            language, serialization_path=self.actual_engine_directory).fit(
            BEVERAGE_DATASET)

        # When
        engine.save()

        # Then
        model_directory_path = os.path.join(self.actual_engine_directory,
                                            "model")

        rule_base_parser_config_path = os.path.join(
            model_directory_path, 'rule_based_parser_config.json')
        mock_rule_based_parser_save.assert_called_once_with(
            rule_base_parser_config_path)

        probabilistic_parser_config_path = os.path.join(
            model_directory_path, 'probabilistic_parser')
        mock_probabilistic_parser_save.assert_called_once_with(
            probabilistic_parser_config_path)

        expected_config_path = os.path.join(self.expected_engine_directory,
                                            "nlu_engine_config.json")
        with io.open(expected_config_path) as f:
            expected_config = json.load(f)

        actual_config_path = os.path.join(self.actual_engine_directory,
                                          "nlu_engine_config.json")
        with io.open(actual_config_path) as f:
            actual_config = json.load(f)

        self.assertDictEqual(actual_config, expected_config)

        expected_model_dir = os.path.join(self.actual_engine_directory,
                                          "model")
        self.assertTrue(os.path.isdir(expected_model_dir))

    def test_should_not_be_saveable_when_no_serialization_path(self):
        # Given
        engine = SnipsNLUEngine(Language.EN).fit(SAMPLE_DATASET)

        # Then
        with self.assertRaises(Exception) as context:
            engine.save()

        error = "A serialization path must be provide to serialize a " \
                "SnipsNLUEngine"
        self.assertTrue(error in context.exception)

    @patch('snips_nlu.nlu_engine.ProbabilisticIntentParser.load')
    @patch('snips_nlu.nlu_engine.RegexIntentParser.load')
    def test_should_be_loadable(self, mock_rule_based_parser_load,
                                mock_probabilistic_parser_load):
        # When
        engine = SnipsNLUEngine.load(self.expected_engine_directory)

        # Then
        expected_intents_data_sizes = {
            "MakeCoffee": 7,
            "MakeTea": 4
        }
        expected_slot_name_mapping = {
            "MakeCoffee": {
                "number_of_cups": "snips/number"
            },
            "MakeTea": {
                "number_of_cups": "snips/number",
                "beverage_temperature": "Temperature"
            }
        }
        expected_entities = {
            "Temperature": {
                "automatically_extensible": True,
                "utterances": {
                    "boiling": "hot",
                    "cold": "cold",
                    "hot": "hot",
                    "iced": "cold"
                }
            }
        }

        rule_based_parser_config_path = os.path.join(
            self.expected_engine_directory, "model",
            "rule_based_parser_config.json")

        mock_rule_based_parser_load.assert_called_once_with(
            rule_based_parser_config_path)

        probabilistic_parser_config_path = os.path.join(
            self.expected_engine_directory, "model", "probabilistic_parser")

        mock_probabilistic_parser_load.assert_called_once_with(
            probabilistic_parser_config_path)

        self.assertEqual(engine.language, Language.EN)
        self.assertDictEqual(engine.intents_data_sizes,
                             expected_intents_data_sizes)
        self.assertEqual(engine.tagging_threshold, 5)
        self.assertDictEqual(engine.slot_name_mapping,
                             expected_slot_name_mapping)
        self.assertDictEqual(engine.entities, expected_entities)

    def test_end_to_end_serialization(self):
        # Given
        dataset = BEVERAGE_DATASET
        engine = SnipsNLUEngine(
            Language.EN, serialization_path=self.actual_engine_directory).fit(
            dataset)
        text = "Give me 3 cups of hot tea please"

        # When
        engine.save()
        engine = SnipsNLUEngine.load(self.actual_engine_directory)
        result = engine.parse(text)

        # Then
        expected_slots = [
            ParsedSlot((8, 9), '3', 'snips/number',
                       'number_of_cups').as_dict(),
            ParsedSlot((18, 21), 'hot', 'Temperature',
                       'beverage_temperature').as_dict()
        ]
        self.assertEqual(result['text'], text)
        self.assertEqual(result['intent']['intent_name'], 'MakeTea')
        self.assertListEqual(result['slots'], expected_slots)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_slots")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser.RegexIntentParser"
           ".get_intent")
    def test_should_handle_keyword_entities(self, mocked_regex_get_intent,
                                            mocked_crf_get_intent,
                                            mocked_crf_get_slots,
                                            mocked_default_features):
        # Given
        language = Language.EN
        dataset = validate_and_format_dataset({
            "snips_nlu_version": "1.1.1",
            "intents": {
                "dummy_intent_1": {
                    ENGINE_TYPE: CUSTOM_ENGINE,
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
            "language": language.iso_code
        })

        mocked_default_features.return_value = []
        mocked_crf_intent = IntentClassificationResult("dummy_intent_1", 1.0)
        mocked_crf_slots = [ParsedSlot(match_range=(0, 7),
                                       value="dummy_3",
                                       entity="dummy_entity_1",
                                       slot_name="dummy_slot_name"),
                            ParsedSlot(match_range=(8, 15),
                                       value="dummy_4",
                                       entity="dummy_entity_2",
                                       slot_name="other_dummy_slot_name")]

        mocked_regex_get_intent.return_value = None
        mocked_crf_get_intent.return_value = mocked_crf_intent
        mocked_crf_get_slots.return_value = mocked_crf_slots

        engine = SnipsNLUEngine(language)
        text = "dummy_3 dummy_4"

        # When
        engine = engine.fit(dataset)
        result = engine.parse(text)

        # Then
        expected_result = Result(
            text, parsed_intent=mocked_crf_intent,
            parsed_slots=[ParsedSlot(match_range=(8, 15), value="dummy_4",
                                     entity="dummy_entity_2",
                                     slot_name="other_dummy_slot_name")]) \
            .as_dict()
        self.assertEqual(result, expected_result)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_slots")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser.RegexIntentParser"
           ".get_intent")
    def test_synonyms_should_point_to_base_value(self, mocked_regex_get_intent,
                                                 mocked_crf_get_intent,
                                                 mocked_crf_get_slots,
                                                 mocked_default_features):
        # Given
        language = Language.EN
        dataset = validate_and_format_dataset({
            "snips_nlu_version": "1.1.1",
            "intents": {
                "dummy_intent_1": {
                    ENGINE_TYPE: CUSTOM_ENGINE,
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
            "language": language.iso_code
        })

        mocked_default_features.return_value = []
        mocked_crf_intent = IntentClassificationResult("dummy_intent_1", 1.0)
        mocked_crf_slots = [ParsedSlot(match_range=(0, 10), value="dummy1_bis",
                                       entity="dummy_entity_1",
                                       slot_name="dummy_slot_name")]

        mocked_regex_get_intent.return_value = None
        mocked_crf_get_intent.return_value = mocked_crf_intent
        mocked_crf_get_slots.return_value = mocked_crf_slots

        engine = SnipsNLUEngine(language).fit(dataset)
        text = "dummy1_bis"

        # When
        result = engine.parse(text)

        # Then
        expected_result = Result(
            text, parsed_intent=mocked_crf_intent,
            parsed_slots=[ParsedSlot(match_range=(0, 10), value="dummy1",
                                     entity="dummy_entity_1",
                                     slot_name="dummy_slot_name")]) \
            .as_dict()
        self.assertEqual(result, expected_result)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.probabilistic_intent_parser"
           ".ProbabilisticIntentParser.get_intent")
    def test_tag_should_return_builtin(
            self, mocked_probabilistic_get_intent,
            mocked_regex_get_intent, mocked_default_features):
        # Given
        mocked_default_features.return_value = []
        mocked_probabilistic_get_intent.return_value = None
        mocked_regex_get_intent.return_value = None

        language = Language.EN
        dataset = validate_and_format_dataset({
            "snips_nlu_version": "1.1.1",
            "intents": {
                "dummy_intent_1": {
                    ENGINE_TYPE: CUSTOM_ENGINE,
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy 1",
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
            "language": language.iso_code
        })
        engine = SnipsNLUEngine(language).fit(dataset)

        # When
        text = "let's meet tomorrow at 3, what do you think?"
        results = engine.tag(text, intent="dummy_intent_1")

        # Then
        expected_results = {
            'intent': {'intent_name': 'dummy_intent_1', 'probability': 1.0},
            'slots': [
                {
                    "range": [11, 24],
                    "value": "tomorrow at 3",
                    "slot_name": "snips/datetime"
                }
            ],
            "text": text
        }

        self.assertEqual(results, expected_results)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_slots")
    @patch("snips_nlu.intent_parser.probabilistic_intent_parser"
           ".ProbabilisticIntentParser.get_intent")
    def test_tag_should_return_custom_entity_over_builtin(
            self, mocked_probabilistic_get_intent, mocked_regex_get_slots,
            mocked_regex_get_intent, mocked_default_features):

        # Given
        intent_name = "dummy_intent_1"
        text = "let's meet tomorrow at 3, what do you think?"
        mocked_default_features.return_value = []
        mocked_probabilistic_get_intent.return_value = None
        mocked_regex_get_intent.return_value = IntentClassificationResult(
            intent_name=intent_name, probability=1.0)
        rng = [6, 24]
        value = "meet tomorrow at 3"
        entity = "my_datetime"
        slot_name = "my_datetime"
        mocked_regex_get_slots.return_value = [ParsedSlot(
            rng, value, entity, slot_name)]

        language = Language.EN
        dataset = validate_and_format_dataset({
            "intents": {
                intent_name: {
                    ENGINE_TYPE: CUSTOM_ENGINE,
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy 1",
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
            "language": language.iso_code,
            "snips_nlu_version": "0.1.1"
        })
        engine = SnipsNLUEngine(language).fit(dataset)

        # When
        results = engine.tag(text, intent=intent_name)

        # Then
        expected_results = {
            'intent': {'intent_name': 'dummy_intent_1', 'probability': 1.0},
            'slots': [
                {
                    "range": [6, 24],
                    "value": "meet tomorrow at 3",
                    "slot_name": slot_name
                }
            ],
            "text": text
        }

        self.assertEqual(results, expected_results)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_slots")
    @patch("snips_nlu.intent_parser.probabilistic_intent_parser"
           ".ProbabilisticIntentParser.get_intent")
    def test_tag_should_tag_seen_entities(
            self, mocked_probabilistic_get_intent, mocked_regex_get_slots,
            mocked_regex_get_intent, mocked_default_features):

        # Given
        intent_name = "dummy_intent_1"
        text = "let's meet tomorrow at 3 with dummy2 bis on time"
        mocked_default_features.return_value = []
        mocked_probabilistic_get_intent.return_value = None
        mocked_regex_get_intent.return_value = IntentClassificationResult(
            intent_name=intent_name, probability=1.0)
        rng = [11, 24]
        value = "tomorrow at 3"
        entity = "my_datetime"

        mocked_regex_get_slots.return_value = [ParsedSlot(
            rng, value, entity, "my_datetime")]

        language = Language.EN
        dataset = validate_and_format_dataset({
            "intents": {
                intent_name: {
                    ENGINE_TYPE: CUSTOM_ENGINE,
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy 1",
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
                                "dummy1 bis"
                            ]
                        }
                    ]
                },
                "dummy_entity_2": {
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "data": [
                        {
                            "value": "dummy2",
                            "synonyms": [
                                "dummy2 bis"
                            ]
                        }
                    ]
                }
            },
            "language": language.iso_code,
            "snips_nlu_version": "0.1.1"
        })
        engine = SnipsNLUEngine(language).fit(dataset)

        # When
        results = engine.tag(text, intent=intent_name)

        # Then
        expected_results = {
            'intent': {'intent_name': 'dummy_intent_1', 'probability': 1.0},
            'slots': [
                {
                    "range": [11, 24],
                    "value": "tomorrow at 3",
                    "slot_name": "my_datetime"
                },
                {
                    "range": [30, 40],
                    "value": "dummy2",
                    "slot_name": "dummy_entity_2"
                }
            ],
            "text": text
        }

        self.assertEqual(results, expected_results)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_slots")
    @patch("snips_nlu.intent_parser.probabilistic_intent_parser"
           ".ProbabilisticIntentParser.get_intent")
    def test_tag_should_return_customs_over_seen_entities(
            self, mocked_probabilistic_get_intent, mocked_regex_get_slots,
            mocked_regex_get_intent, mocked_default_features):

        # Given
        intent_name = "dummy_intent_1"
        text = "let's meet tomorrow at 3 with dummy2 bis on time"
        mocked_default_features.return_value = []
        mocked_probabilistic_get_intent.return_value = None
        mocked_regex_get_intent.return_value = IntentClassificationResult(
            intent_name=intent_name, probability=1.0)
        rng = [30, 43]
        value = "dummy2 bis on"
        entity = "my_datetime"

        mocked_regex_get_slots.return_value = [ParsedSlot(
            rng, value, entity, "my_datetime")]

        language = Language.EN
        dataset = validate_and_format_dataset({
            "intents": {
                intent_name: {
                    ENGINE_TYPE: CUSTOM_ENGINE,
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy 1",
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
                                "dummy1 bis"
                            ]
                        }
                    ]
                },
                "dummy_entity_2": {
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "data": [
                        {
                            "value": "dummy2",
                            "synonyms": [
                                "dummy2 bis"
                            ]
                        }
                    ]
                }
            },
            "language": language.iso_code,
            "snips_nlu_version": "0.1.1"
        })
        engine = SnipsNLUEngine(language).fit(dataset)

        # When
        results = engine.tag(text, intent=intent_name)

        # Then
        expected_results = {
            'intent': {'intent_name': 'dummy_intent_1', 'probability': 1.0},
            'slots': [
                {
                    "range": [11, 24],
                    "value": "tomorrow at 3",
                    "slot_name": "snips/datetime"
                },
                {
                    "range": [30, 43],
                    "value": "dummy2 bis on",
                    "slot_name": "my_datetime"
                }
            ],
            "text": text
        }

        self.assertEqual(results, expected_results)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser"
           ".RegexIntentParser.get_slots")
    @patch("snips_nlu.intent_parser.probabilistic_intent_parser"
           ".ProbabilisticIntentParser.get_intent")
    def test_tag_should_not_return_ambiguous_seen_entities(
            self, mocked_probabilistic_get_intent, mocked_regex_get_slots,
            mocked_regex_get_intent, mocked_default_features):

        # Given
        intent_name = "dummy_intent_1"
        text = "let's meet tomorrow at 3 with dummy2 bis on time"
        mocked_default_features.return_value = []
        mocked_probabilistic_get_intent.return_value = None
        mocked_regex_get_intent.return_value = IntentClassificationResult(
            intent_name=intent_name, probability=1.0)
        rng = [25, 29]
        value = "with"
        entity = "my_datetime"

        mocked_regex_get_slots.return_value = [ParsedSlot(
            rng, value, entity, "my_datetime")]

        language = Language.EN
        dataset = validate_and_format_dataset({
            "intents": {
                intent_name: {
                    ENGINE_TYPE: CUSTOM_ENGINE,
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy 1",
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
                            "value": "dummy2",
                            "synonyms": [
                                "dummy2",
                            ]
                        }
                    ]
                },
                "dummy_entity_2": {
                    "use_synonyms": True,
                    "automatically_extensible": False,
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
            "language": language.iso_code,
            "snips_nlu_version": "0.1.1"
        })
        engine = SnipsNLUEngine(language).fit(dataset)

        # When
        results = engine.tag(text, intent=intent_name)

        # Then
        expected_results = {
            'intent': {'intent_name': 'dummy_intent_1', 'probability': 1.0},
            'slots': [
                {
                    "range": [11, 24],
                    "value": "tomorrow at 3",
                    "slot_name": "snips/datetime"
                },
                {
                    "range": [25, 29],
                    "value": "with",
                    "slot_name": "my_datetime"
                }
            ],
            "text": text
        }

        self.assertEqual(results, expected_results)

    def test_enrich_slots(self):
        # Given
        slots = [
            # Adjacent
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((2, 6), "", "", ""),
                    ParsedSlot((8, 10), "", "", "")
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", ""),
                    ParsedSlot((2, 6), "", "", ""),
                    ParsedSlot((8, 10), "", "", "")
                ]
            },
            # Equality
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((6, 8), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Inclusion
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((5, 7), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Cross upper
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((7, 10), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Cross lower
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((5, 7), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Full overlap
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((4, 12), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            }
        ]

        for data in slots:
            # When
            enriched = enrich_slots(data["slots"], data["other_slots"])

            # Then
            self.assertEqual(enriched, data["enriched"])

    def test_should_parse_naughty_strings(self):
        # Given
        dataset = SAMPLE_DATASET
        naughty_strings_path = os.path.join(TEST_PATH, "resources",
                                            "naughty_strings.txt")
        with io.open(naughty_strings_path, encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        # When
        engine = SnipsNLUEngine(Language.EN).fit(dataset)

        # Then
        for s in naughty_strings:
            try:
                engine.parse(s)
            except Exception, e:
                self.fail('Exception raised: %s' % e.message)

    def test_should_fit_with_naughty_strings(self):
        # Given
        naughty_strings_path = os.path.join(TEST_PATH, "resources",
                                            "naughty_strings.txt")
        with io.open(naughty_strings_path, encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]
        utterances = [{DATA: [{TEXT: naughty_string}]} for naughty_string in
                      naughty_strings]

        # When
        naughty_dataset = validate_and_format_dataset({
            "intents": {
                "naughty_intent": {
                    "engineType": "regex",
                    "utterances": utterances
                }
            },
            "entities": dict(),
            "language": "en",
            "snips_nlu_version": "0.0.1"
        })

        # Then
        error = None
        raised = False
        try:
            SnipsNLUEngine(Language.EN).fit(naughty_dataset)
        except Exception, e:
            raised = True
            error = e
        self.assertFalse(raised, 'Exception raised: %s' % str(error))

    def test_engine_should_fit_with_builtins_entities(self):
        # Given
        language = Language.EN
        dataset = validate_and_format_dataset({
            "intents": {
                "dummy": {
                    ENGINE_TYPE: CUSTOM_ENGINE,
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
            "language": language.iso_code,
            "snips_nlu_version": "0.0.1"
        })

        # When / Then
        # noinspection PyBroadException
        try:
            SnipsNLUEngine(language).fit(dataset)
        except:
            self.fail("NLU engine should fit builtin")
