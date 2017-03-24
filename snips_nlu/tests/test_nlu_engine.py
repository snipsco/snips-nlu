import unittest

from mock import Mock

from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
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
