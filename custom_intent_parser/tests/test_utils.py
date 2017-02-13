import unittest

from custom_intent_parser.utils import LimitedSizeDict


class TestLimitedSizeDict(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
