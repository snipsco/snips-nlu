from __future__ import unicode_literals

import unittest

from snips_nlu.languages import Language
from snips_nlu.resources import RESOURCE_INDEX, get_stop_words


class TestResources(unittest.TestCase):
    def test_resources_index_should_have_all_languages(self):
        # Given
        index = RESOURCE_INDEX

        # When
        languages = index.keys()

        # Then
        self.assertEqual(len(languages), len(Language.__members__))

    def test_all_languages_should_have_stop_words(self):
        # The capitalization for the CRF assumes all languages have stop_words
        # Given
        languages = Language

        for l in languages:
            try:
                # When/Then
                get_stop_words(l)
            except:
                self.fail("%s has not stop words" % l.iso_code)
