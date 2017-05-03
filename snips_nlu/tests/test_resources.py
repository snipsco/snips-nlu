import unittest

from snips_nlu.languages import Language
from snips_nlu.resources import RESOURCE_INDEX


class TestResources(unittest.TestCase):
    def test_resources_index_should_have_all_languages(self):
        # Given
        index = RESOURCE_INDEX

        # When
        languages = index.keys()

        # Then
        self.assertEqual(len(languages), len(Language.__members__))
