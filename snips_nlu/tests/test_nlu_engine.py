import unittest

from mock import Mock

from snips_nlu.nlu_engine import SnipsNLUEngine
from ..result import Result, ParsedEntity, IntentClassificationResult


class TestSnipsNLUEngine(unittest.TestCase):
    def test_should_parse_with_custom_parsers_first(self):
        # Given
        mocked_parser1 = Mock()
        intent_result1 = IntentClassificationResult(
            intent_name='mocked_intent1', probability=0.5)
        intent_entities1 = []
        mocked_parser1.get_intent.return_value = intent_result1
        mocked_parser1.get_entities.return_value = intent_entities1

        mocked_parser2 = Mock()
        intent_result2 = IntentClassificationResult(
            intent_name='mocked_intent2', probability=0.7)
        intent_entities2 = [
            ParsedEntity(match_range=(3, 5), value='mocked_value',
                         entity='mocked_entity', slot_name='mocked_slot_name')]
        mocked_parser2.get_intent.return_value = intent_result2
        mocked_parser2.get_entities.return_value = intent_entities2

        mocked_builtin_parser = Mock()
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_entities.return_value = builtin_entities

        engine = SnipsNLUEngine([mocked_parser1, mocked_parser2],
                                mocked_builtin_parser)

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, intent_result2, intent_entities2))

    def test_should_parse_with_builtin_when_no_custom(self):
        # When
        mocked_builtin_parser = Mock()
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_entities.return_value = builtin_entities
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
        mocked_parser1.get_entities.return_value = []

        mocked_parser2 = Mock()
        mocked_parser2.get_intent.return_value = IntentClassificationResult(
            intent_name='mocked_custom_intent', probability=0.)
        mocked_parser2.get_entities.return_value = []

        mocked_builtin_parser = Mock()
        builtin_intent_result = IntentClassificationResult(
            intent_name='mocked_builtin_intent', probability=0.9)
        builtin_entities = []
        mocked_builtin_parser.get_intent.return_value = builtin_intent_result
        mocked_builtin_parser.get_entities.return_value = builtin_entities

        engine = SnipsNLUEngine([mocked_parser1, mocked_parser2],
                                mocked_builtin_parser)

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, builtin_intent_result,
                                       builtin_entities))

    def test_should_not_fail_when_no_custom_no_builtin(self):
        # Given
        engine = SnipsNLUEngine()

        # When
        text = "hello world"
        parse = engine.parse(text)

        # Then
        self.assertEqual(parse, Result(text, None, None))
