from __future__ import unicode_literals

import unittest

from snips_nlu_ontology import get_all_languages

from snips_nlu.resources import RESOURCE_INDEX, get_stop_words


class TestResources(unittest.TestCase):
    def test_resources_index_should_have_all_languages(self):
        # Given
        index = RESOURCE_INDEX

        # When
        languages = set(index)

        # Then
        self.assertSetEqual(languages, get_all_languages())

    def test_all_languages_should_have_stop_words(self):
        # The capitalization for the CRF assumes all languages have stop_words
        # Given
        for language in get_all_languages():
            try:
                # When/Then
                get_stop_words(language)
            except:  # pylint: disable=W0702
                self.fail("%s has not stop words" % language)
