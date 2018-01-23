import unittest
from future.utils import iteritems
from snips_nlu.utils import LimitedSizeDict, ranges_overlap


class TestLimitedSizeDict(unittest.TestCase):
    def test_should_raise_when_no_size_limit(self):
        # Given/When/Then
        with self.assertRaises(ValueError) as ctx:
            LimitedSizeDict()
        self.assertEqual(str(ctx.exception.args[0]),
                         "'size_limit' must be passed as a keyword argument")

    def test_should_initialize_with_argument(self):
        # Given
        sequence = [("a", 1), ("b", 2)]
        size_limit = 3
        # When
        d = LimitedSizeDict(sequence, size_limit=size_limit)
        # Then
        items = sorted(iteritems(d), key=lambda i: i[0])
        self.assertListEqual(items, sequence)

    def test_should_initialize_without_argument(self):
        # Given
        size_limit = 10
        # When
        d = LimitedSizeDict(size_limit=size_limit)
        # Then
        self.assertListEqual(list(d), [])

    def test_should_wrong_when_initialization_should_raise_error(self):
        # Given
        sequence = [("a", 1), ("b", 2), ("c", 3)]
        size_limit = len(sequence) - 1
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            LimitedSizeDict(sequence, size_limit=size_limit)
        self.assertEqual(str(ctx.exception.args[0]),
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
        items = sorted(list(iteritems(my_dict)), key=lambda i: i[0])
        self.assertListEqual(items, sequence[size_limit:])


class TestUtils(unittest.TestCase):
    def test_ranges_overlap(self):
        # Given
        range1 = [4, 8]
        range2 = [5, 7]
        range3 = [3, 9]
        range4 = [3, 4]
        range5 = [8, 9]
        range6 = [3, 6]
        range7 = [4, 10]

        # When / Then
        self.assertTrue(ranges_overlap(range1, range2))
        self.assertTrue(ranges_overlap(range1, range3))
        self.assertFalse(ranges_overlap(range1, range4))
        self.assertFalse(ranges_overlap(range1, range5))
        self.assertTrue(ranges_overlap(range1, range6))
        self.assertTrue(ranges_overlap(range1, range7))
