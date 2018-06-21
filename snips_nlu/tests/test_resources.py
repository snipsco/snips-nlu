from __future__ import unicode_literals

import unittest

from mock import patch

from snips_nlu.constants import DATA_PATH
from snips_nlu.resources import _get_resource, MissingResource, \
    clear_resources, load_resources, resource_exists


class TestResources(unittest.TestCase):
    def test_should_load_resources_from_data_path(self):
        # Given
        clear_resources()

        # When
        load_resources("en")

        # Then
        self.assertTrue(resource_exists("en", "gazetteers"))

    def test_should_load_resources_from_package(self):
        # Given
        clear_resources()

        # When
        load_resources("snips_nlu_en")

        # Then
        self.assertTrue(resource_exists("en", "gazetteers"))

    def test_should_load_resources_from_path(self):
        # Given
        clear_resources()
        resources_path = DATA_PATH / "en"

        # When
        load_resources(str(resources_path))

        # Then
        self.assertTrue(resource_exists("en", "gazetteers"))

    def test_should_fail_loading_unknown_resources(self):
        # Given
        unknown_resource_name = "foobar"

        # When / Then
        with self.assertRaises(MissingResource):
            load_resources(unknown_resource_name)

    def test_should_raise_missing_resource_when_language_not_found(self):
        # Given
        mocked_value = dict()

        # When
        with patch("snips_nlu.resources._RESOURCES", mocked_value):
            with self.assertRaises(MissingResource):
                _get_resource("en", "foobar")

    def test_should_raise_missing_resource_when_resource_not_found(self):
        # Given
        mocked_value = {"en": dict()}

        # When
        with patch("snips_nlu.resources._RESOURCES", mocked_value):
            with self.assertRaises(MissingResource):
                _get_resource("en", "foobar")
