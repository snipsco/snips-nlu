import unittest

from snips_nlu.utils import LimitedSizeDict
from snips_nlu.result import Result, ParsedSlot, IntentClassificationResult


class TestLimitedSizeDict(unittest.TestCase):
    def test_should_raise_when_no_size_limit(self):
        # Given/When/Then
        with self.assertRaises(ValueError) as ctx:
            LimitedSizeDict()
        self.assertEqual(ctx.exception.message,
                         "'size_limit' must be passed as a keyword argument")

    def test_should_initialize_with_argument(self):
        # Given
        sequence = [("a", 1), ("b", 2)]
        size_limit = 3
        # When
        d = LimitedSizeDict(sequence, size_limit=size_limit)
        # Then
        self.assertItemsEqual(d.items(), sequence)

    def test_should_initialize_without_argument(self):
        # Given
        size_limit = 10
        # When
        d = LimitedSizeDict(size_limit=size_limit)
        # Then
        self.assertItemsEqual(d.items(), [])

    def test_should_wrong_when_initialization_should_raise_error(self):
        # Given
        sequence = [("a", 1), ("b", 2), ("c", 3)]
        size_limit = len(sequence) - 1
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            LimitedSizeDict(sequence, size_limit=size_limit)
        self.assertEqual(ctx.exception.message,
                         "Tried to initialize LimitedSizedDict with more "
                         "value than permitted with 'limit_size'")

    def test_should_erase_items_when_updating(self):
        # Given
        sequence = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        size_limit = len(sequence) - 2
        # When
        my_dict = LimitedSizeDict(sequence[:size_limit], size_limit=size_limit)
        for k, v in sequence[size_limit:]:
            my_dict[k] = v
        # Then
        self.assertItemsEqual(my_dict.items(), sequence[size_limit:])

    def test_should_serialize_results(self):
        # Given
        result = Result(text="hello world",
                        parsed_intent=IntentClassificationResult("world", 0.5),
                        parsed_slots=[
                            ParsedSlot((3, 5), "slot_value", "slot_entity",
                                       "slot_name")])
        # When
        result_dict = result.as_dict()

        # Then
        expected_dict = {
            'parsed_intent': {'intent_name': 'world', 'probability': 0.5},
            'parsed_slots': [{'match_range': [3, 5],
                              'slot_name': 'slot_name',
                              'value': 'slot_value'}],
            'text': 'hello world'}
        self.assertDictEqual(result_dict, expected_dict)

    def test_should_serialize_results_when_none_values(self):
        # Given
        result = Result(text="hello world", parsed_intent=None,
                        parsed_slots=None)

        # When
        result_dict = result.as_dict()

        # Then
        expected_dict = {
            'parsed_intent': None,
            'parsed_slots': None,
            'text': 'hello world'}
        self.assertDictEqual(result_dict, expected_dict)


if __name__ == '__main__':
    unittest.main()
