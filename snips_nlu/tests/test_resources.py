from __future__ import unicode_literals

from mock import patch
from snips_nlu_ontology import get_all_languages

from snips_nlu.resources import RESOURCE_INDEX, get_stop_words, get_resource, \
    UnloadedResources, UnknownResource
from snips_nlu.tests.utils import SnipsTest


class TestResources(SnipsTest):
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

    def test_should_raise_unloaded_resources(self):
        # Given
        mocked_value = dict()

        # When
        with patch("snips_nlu.resources._RESOURCES", mocked_value):
            with self.assertRaises(UnloadedResources) as ctx:
                get_resource("en", "")
        self.assertEqual(
            ctx.exception.message,
            "Missing resources for 'en', please load them with the "
            "load_resources function")

    def test_should_raise_non_existing_resources(self):
        # Given
        mocked_value = {"en": dict()}

        # When
        with patch("snips_nlu.resources._RESOURCES", mocked_value):
            with self.assertRaises(UnknownResource) as ctx:
                get_resource("en", "my_resource")
        self.assertEqual(
            ctx.exception.message,
            "Unknown resource 'my_resource' for 'en' language")
