import json
import unittest

import numpy as np
from mock import Mock, patch, call

from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.result import Result, ParsedSlot, IntentClassificationResult
from snips_nlu.slot_filler.feature_functions import BaseFeatureFunction
from utils import SAMPLE_DATASET


def mocked_default(_, use_stemming):
    return []


class TestSnipsNLUEngine(unittest.TestCase):
    def test_should_use_parsers_sequentially(self):
        # Given
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

        mocked_builtin_parser = Mock()
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_slots.return_value = builtin_entities

        engine = SnipsNLUEngine()
        engine.custom_parsers = [mocked_parser1, mocked_parser2]
        engine.builtin_parser = mocked_builtin_parser
        engine.entities = {"mocked_entity": {"automatically_extensible": True}}

        # When
        parse = engine.parse(input_text)

        # Then
        self.assertEqual(parse,
                         Result(input_text, intent_result2,
                                intent_entities2).as_dict())

    def test_should_parse_with_builtin_when_no_custom(self):
        # When
        mocked_builtin_parser = Mock()
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_slots.return_value = builtin_entities
        engine = SnipsNLUEngine()
        engine.builtin_parser = mocked_builtin_parser

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse,
                         Result(text, builtin_intent_result,
                                builtin_entities).as_dict())

    def test_should_parse_with_builtin_when_customs_return_nothing(self):
        # Given
        mocked_parser1 = Mock()
        mocked_parser1.get_intent.return_value = None
        mocked_parser1.get_slots.return_value = []

        mocked_parser2 = Mock()
        mocked_parser2.get_intent.return_value = None
        mocked_parser2.get_slots.return_value = []

        mocked_builtin_parser = Mock()
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_slots.return_value = builtin_entities

        engine = SnipsNLUEngine()
        engine.custom_parsers = [mocked_parser1, mocked_parser2]
        engine.builtin_parser = mocked_builtin_parser

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, builtin_intent_result,
                                       builtin_entities).as_dict())

    def test_should_not_fail_when_no_parsers(self):
        # Given
        engine = SnipsNLUEngine()

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, None, None).as_dict())

    def test_should_be_serializable(self):
        # Given
        engine = SnipsNLUEngine().fit(SAMPLE_DATASET)
        text = "this is a dummy_1 query with another dummy_2"
        expected_parse = engine.parse(text)

        # When
        serialized_engine = engine.to_dict()
        deserialized_engine = SnipsNLUEngine.load_from(
            language='en',
            customs=serialized_engine)

        # Then
        try:
            dumped = json.dumps(serialized_engine).decode("utf8")
        except:
            self.fail("NLU engine dict should be json serializable to utf8")

        try:
            _ = SnipsNLUEngine.from_dict(json.loads(dumped))
        except:
            self.fail("SnipsNLUEngine should be deserializable from dict with "
                      "unicode values")

        self.assertEqual(deserialized_engine.parse(text), expected_parse)

    @patch("snips_nlu.slot_filler.feature_functions.default_features",
           side_effect=mocked_default)
    @patch("snips_nlu.slot_filler.feature_functions.get_token_is_in")
    def test_should_add_custom_entity_in_collection_feature(
            self, mocked_get_token, mocked_default_features):
        np.random.seed(1)

        # Given
        def mocked_get_token_is_in(collection, collection_name,
                                   use_stemming=False):
            def f(index, cache):
                return None

            return BaseFeatureFunction("token_is_in_%s" % collection_name, f)

        mocked_get_token.side_effect = mocked_get_token_is_in

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
                                    "text": " query."
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "2 P.M",
                                    "entity": "snips/datetime",
                                    "slot_name": "origin_time"
                                },
                                {
                                    "text": "dummy_2",
                                    "entity": "dummy_entity_2",
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

        # When
        SnipsNLUEngine().fit(dataset)

        np.random.seed(1)
        keep_prob = .5
        collection_1 = ["dummy1", "dummy1_bis", "dummy2", "dummy2_bis"]
        length_collection_1 = int(keep_prob * len(collection_1))
        collection_1 = np.random.choice(collection_1, length_collection_1,
                                        replace=False).tolist()

        collection_2 = ["dummy2"]

        # Then
        calls = [
            call(collection=collection_1,
                 collection_name="dummy_entity_1", use_stemming=False),
            call(collection=collection_2, collection_name="dummy_entity_2",
                 use_stemming=False),
        ]

        mocked_get_token.assert_has_calls(calls, any_order=True)

    @patch("snips_nlu.slot_filler.feature_functions.default_features",
           side_effect=mocked_default)
    @patch("snips_nlu.intent_parser.crf_intent_parser.CRFIntentParser"
           ".get_slots")
    @patch("snips_nlu.intent_parser.crf_intent_parser.CRFIntentParser"
           ".get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser.RegexIntentParser"
           ".get_intent")
    def test_should_handle_keyword_entities(self, mocked_regex_get_intent,
                                            mocked_crf_get_intent,
                                            mocked_crf_get_slots, _):
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

        def mocked_regex_intent(_):
            return None

        def mocked_crf_intent(_):
            return IntentClassificationResult("dummy_intent_1", 1.0)

        def mocked_crf_slots(_, intent=None):
            return [ParsedSlot(match_range=(0, 7),
                               value="dummy_3",
                               entity="dummy_entity_1",
                               slot_name="dummy_slot_name"),
                    ParsedSlot(match_range=(8, 15),
                               value="dummy_4",
                               entity="dummy_entity_2",
                               slot_name="other_dummy_slot_name")]

        mocked_regex_get_intent.side_effect = mocked_regex_intent
        mocked_crf_get_intent.side_effect = mocked_crf_intent
        mocked_crf_get_slots.side_effect = mocked_crf_slots

        engine = SnipsNLUEngine()
        text = "dummy_3 dummy_4"

        # When
        engine = engine.fit(dataset)
        result = engine.parse(text)

        # Then
        expected_result = Result(
            text, parsed_intent=mocked_crf_intent(text),
            parsed_slots=[ParsedSlot(match_range=(8, 15), value="dummy_4",
                                     entity="dummy_entity_2",
                                     slot_name="other_dummy_slot_name")]) \
            .as_dict()
        self.assertEqual(result, expected_result)

    @patch("snips_nlu.slot_filler.feature_functions.default_features",
           side_effect=mocked_default)
    @patch("snips_nlu.intent_parser.crf_intent_parser.CRFIntentParser"
           ".get_slots")
    @patch("snips_nlu.intent_parser.crf_intent_parser.CRFIntentParser"
           ".get_intent")
    @patch("snips_nlu.intent_parser.regex_intent_parser.RegexIntentParser"
           ".get_intent")
    def test_synonyms_should_point_to_base_value(self, mocked_regex_get_intent,
                                                 mocked_crf_get_intent,
                                                 mocked_crf_get_slots, _):
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

        def mocked_regex_intent(_):
            return None

        def mocked_crf_intent(_):
            return IntentClassificationResult("dummy_intent_1", 1.0)

        def mocked_crf_slots(_, intent=None):
            return [ParsedSlot(match_range=(0, 10), value="dummy1_bis",
                               entity="dummy_entity_1",
                               slot_name="dummy_slot_name")]

        mocked_regex_get_intent.side_effect = mocked_regex_intent
        mocked_crf_get_intent.side_effect = mocked_crf_intent
        mocked_crf_get_slots.side_effect = mocked_crf_slots

        engine = SnipsNLUEngine().fit(dataset)
        text = "dummy1_bis"

        # When
        result = engine.parse(text)

        # Then
        expected_result = Result(
            text, parsed_intent=mocked_crf_intent(text),
            parsed_slots=[ParsedSlot(match_range=(0, 10), value="dummy1",
                                     entity="dummy_entity_1",
                                     slot_name="dummy_slot_name")]) \
            .as_dict()
        self.assertEqual(result, expected_result)
