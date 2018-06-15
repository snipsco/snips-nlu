from __future__ import unicode_literals

import unittest

from mock import patch

from snips_nlu.resources import _get_resource, MissingResource


class TestResources(unittest.TestCase):
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
