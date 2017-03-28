import unittest

from mock import Mock, patch, call

from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.result import Result, ParsedSlot, IntentClassificationResult
from utils import SAMPLE_DATASET



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

        engine = SnipsNLUEngine([mocked_parser1, mocked_parser2],
                                mocked_builtin_parser)

        # When
        parse = engine.parse(input_text)

        # Then
        self.assertEqual(parse,
                         Result(input_text, intent_result2, intent_entities2))

    def test_should_parse_with_builtin_when_no_custom(self):
        # When
        mocked_builtin_parser = Mock()
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_slots.return_value = builtin_entities
        engine = SnipsNLUEngine([], mocked_builtin_parser)

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse,
                         Result(text, builtin_intent_result, builtin_entities))

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

        engine = SnipsNLUEngine([mocked_parser1, mocked_parser2],
                                mocked_builtin_parser)

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, builtin_intent_result,
                                       builtin_entities))

    def test_should_not_fail_when_no_parsers(self):
        # Given
        engine = SnipsNLUEngine()

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, None, None))

    def test_should_be_serializable(self):
        # Given
        engine = SnipsNLUEngine().fit(SAMPLE_DATASET)
        text = "this is a dummy_1 query with another dummy_2"

        # When
        serialized_engine = engine.to_dict()
        deserialized_engine = SnipsNLUEngine.from_dict(serialized_engine)
        expected_parse = engine.parse(text)

        # Then
        parse = deserialized_engine.parse(text)
        self.assertEqual(parse, expected_parse)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch("snips_nlu.slot_filler.feature_functions.get_token_is_in")
    def test_should_add_custom_entity_in_collection_feature(
            self, mocked_get_token, mocked_default_features):
        # Given
        def mocked_get_token_is_in(collection, entity_name):
            return (entity_name, lambda tokens, token_index: None)

        def mocked_default(language):
            return []

        mocked_get_token.side_effect = mocked_get_token_is_in
        mocked_default_features.side_effect = mocked_default

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
            }
        }

        # When
        SnipsNLUEngine().fit(dataset)

        # Then
        calls = [
            call(["dummy1", "dummy1_bis", "dummy2", "dummy2_bis"],
                 "dummy_entity_1"),
            call(["dummy2"], "dummy_entity_2")
        ]

        mocked_get_token.assert_has_calls(calls, any_order=True)
