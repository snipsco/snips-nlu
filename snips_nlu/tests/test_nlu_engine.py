from __future__ import unicode_literals

import io
import json
import os
import unittest

from mock import Mock, patch

from snips_nlu.constants import ENGINE_TYPE, CUSTOM_ENGINE, DATA, TEXT
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.result import Result, ParsedSlot, IntentClassificationResult
from utils import SAMPLE_DATASET, empty_dataset, TEST_PATH


class TestSnipsNLUEngine(unittest.TestCase):
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

        mocked_builtin_parser = Mock(parser=Mock(language=language.iso_code))

        builtin_intent_result = None
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_slots.return_value = builtin_entities

        mocked_entities = {"mocked_entity": {"automatically_extensible": True}}
        engine = SnipsNLUEngine(
            language, entities=mocked_entities,
            rule_based_parser=mocked_parser1,
            probabilistic_parser=mocked_parser2,
            builtin_parser=mocked_builtin_parser,
            slot_name_mapping={'mocked_slot_name': 'mocked_entity'})

        # When
        parse = engine.parse(input_text)

        # Then
        expected_parse = Result(input_text, intent_result2,
                                intent_entities2).as_dict()
        self.assertEqual(parse, expected_parse)

    def test_should_parse_with_builtin_when_no_custom(self):
        # When
        language = Language.EN
        mocked_builtin_parser = Mock(parser=Mock(language=language.iso_code))
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_slots.return_value = builtin_entities
        engine = SnipsNLUEngine(language, builtin_parser=mocked_builtin_parser)

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse,
                         Result(text, builtin_intent_result,
                                builtin_entities).as_dict())

    def test_should_parse_with_builtin_when_customs_return_nothing(self):
        # Given
        language = Language.EN
        mocked_parser1 = Mock()
        mocked_parser1.get_intent.return_value = None
        mocked_parser1.get_slots.return_value = []

        mocked_parser2 = Mock()
        mocked_parser2.get_intent.return_value = None
        mocked_parser2.get_slots.return_value = []

        mocked_builtin_parser = Mock(parser=Mock(language=language.iso_code))
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_slots.return_value = builtin_entities

        engine = SnipsNLUEngine(
            language, builtin_parser=mocked_builtin_parser,
            rule_based_parser=mocked_parser1,
            probabilistic_parser=mocked_parser2)

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, builtin_intent_result,
                                       builtin_entities).as_dict())

    def test_should_handle_empty_dataset(self):
        # Given
        engine = SnipsNLUEngine(Language.EN).fit(empty_dataset(Language.EN))

        # When
        result = engine.parse("hello world")

        # Then
        self.assertEqual(result, Result("hello world", None, None).as_dict())

    def test_should_be_serializable(self):
        # Given
        language = Language.EN
        engine = SnipsNLUEngine(language).fit(SAMPLE_DATASET)
        text = "this is a dummy_1 query with another dummy_2"
        expected_parse = engine.parse(text)

        # When
        serialized_engine = engine.to_dict()
        deserialized_engine = SnipsNLUEngine.load_from(
            language=language.iso_code,
            customs=serialized_engine)

        # Then
        # noinspection PyBroadException
        try:
            dumped = json.dumps(serialized_engine).decode("utf8")
        except:
            self.fail("NLU engine dict should be json serializable to utf8")

        # noinspection PyBroadException
        try:
            _ = SnipsNLUEngine.load_from(language=language.iso_code,
                                         customs=json.loads(dumped))
        except:
            self.fail("SnipsNLUEngine should be deserializable from dict with "
                      "unicode values")

        self.assertEqual(deserialized_engine.parse(text), expected_parse)

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
    def test_ui_parse_should_return_builtin(
            self, mocked_probabilistic_get_intent,
            mocked_regex_get_intent, mocked_default_features):
        # Given
        mocked_default_features.return_value = []
        mocked_probabilistic_get_intent.return_value = None
        mocked_regex_get_intent.return_value = None

        language = Language.EN
        dataset = validate_and_format_dataset({
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
        results = engine.parse(text, intent="dummy_intent_1",
                               force_builtin_entities=True)

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
    def test_parse_with_builtin_force_should_return_custom_when_overlapping(
            self, mocked_probabilistic_get_intent, mocked_regex_get_slots,
            mocked_regex_get_intent, mocked_default_features):

        # Given
        intent_name = "dummy_intent_1"
        text = "let's meet tomorrow at 3, what do you think?"
        mocked_default_features.return_value = []
        mocked_probabilistic_get_intent.return_value = None
        mocked_regex_get_intent.return_value = IntentClassificationResult(
            intent_name=intent_name, probability=1.0)
        range = [11, 24]
        value = "tomorrow at 3"
        entity = "my_datetime"
        slot_name = "my_datetime"
        mocked_regex_get_slots.return_value = [ParsedSlot(
            range, value, entity, slot_name)]

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
            "language": language.iso_code
        })
        engine = SnipsNLUEngine(language).fit(dataset)

        # When
        results = engine.parse(text, intent=intent_name,
                               force_builtin_entities=True)

        # Then
        expected_results = {
            'intent': {'intent_name': 'dummy_intent_1', 'probability': 1.0},
            'slots': [
                {
                    "range": [11, 24],
                    "value": "tomorrow at 3",
                    "slot_name": slot_name
                }
            ],
            "text": text
        }

        self.assertEqual(results, expected_results)

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
            raised = False
            error = None
            try:
                engine.parse(s)
            except Exception, e:
                raised = True
                error = e
            self.assertFalse(raised, 'Exception raised: %s' % str(error))

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
            "language": "en"
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
            "language": language.iso_code
        })

        # When / Then
        try:
            SnipsNLUEngine(language).fit(dataset)
        except:
            self.fail("NLU engine should fit builtin")
