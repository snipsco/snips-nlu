from snips_nlu.tests.utils import SnipsTest
from snips_nlu.utils import ranges_overlap


class TestUtils(SnipsTest):
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
