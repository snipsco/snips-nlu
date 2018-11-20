from __future__ import unicode_literals

import io
from unittest import TestCase

import yaml
from deprecation import fail_if_not_removed
from mock import patch

from snips_nlu.dataset import Entity, EntityFormatError


class TestEntityLoading(TestCase):
    def test_from_yaml_file(self):
        # Given
        yaml_stream = io.StringIO("""
# Location Entity
---
type: entity
name: location
automatically_extensible: no
use_synonyms: yes
matching_strictness: 0.5
values:
- [new york, big apple]
- [paris, city of lights]
- london
        """)
        yaml_dict = yaml.safe_load(yaml_stream)

        # When
        entity = Entity.from_yaml(yaml_dict)
        entity_dict = entity.json

        # Then
        expected_entity_dict = {
            "automatically_extensible": False,
            "data": [
                {
                    "synonyms": [
                        "big apple"
                    ],
                    "value": "new york"
                },
                {
                    "synonyms": [
                        "city of lights"
                    ],
                    "value": "paris"
                },
                {
                    "synonyms": [],
                    "value": "london"
                }
            ],
            "use_synonyms": True,
            "matching_strictness": 0.5
        }
        self.assertDictEqual(expected_entity_dict, entity_dict)

    def test_from_yaml_file_with_defaults(self):
        # Given
        yaml_stream = io.StringIO("""
# Location Entity
---
name: location
values:
- [new york, big apple]
- [paris, city of lights]
- london
        """)
        yaml_dict = yaml.safe_load(yaml_stream)

        # When
        entity = Entity.from_yaml(yaml_dict)
        entity_dict = entity.json

        # Then
        expected_entity_dict = {
            "automatically_extensible": True,
            "data": [
                {
                    "synonyms": [
                        "big apple"
                    ],
                    "value": "new york"
                },
                {
                    "synonyms": [
                        "city of lights"
                    ],
                    "value": "paris"
                },
                {
                    "synonyms": [],
                    "value": "london"
                }
            ],
            "use_synonyms": True,
            "matching_strictness": 1.0
        }
        self.assertDictEqual(expected_entity_dict, entity_dict)

    def test_fail_from_yaml_file_when_wrong_type(self):
        # Given
        yaml_stream = io.StringIO("""
# Location Entity
---
type: intent
name: location
values:
- [new york, big apple]
- [paris, city of lights]
- london
        """)
        yaml_dict = yaml.safe_load(yaml_stream)

        # When / Then
        with self.assertRaises(EntityFormatError):
            Entity.from_yaml(yaml_dict)

    def test_fail_from_yaml_file_when_no_name(self):
        # Given
        yaml_stream = io.StringIO("""
# Location Entity
---
values:
- [new york, big apple]
- [paris, city of lights]
- london
        """)
        yaml_dict = yaml.safe_load(yaml_stream)

        # When / Then
        with self.assertRaises(EntityFormatError):
            Entity.from_yaml(yaml_dict)

    @patch("pathlib.io")
    @fail_if_not_removed
    def test_from_text_file(self, mock_io):
        # Given
        entity_file = "entity_location.txt"
        location_txt = """
new york,big apple
paris,city of lights
london
        """

        # pylint:disable=unused-argument
        def mock_open(self_, *args, **kwargs):
            if str(self_) == entity_file:
                return io.StringIO(location_txt)
            return None

        # pylint:enable=unused-argument
        mock_io.open.side_effect = mock_open

        # When
        entity = Entity.from_file(entity_file)
        entity_dict = entity.json

        # Then
        expected_entity_dict = {
            "automatically_extensible": True,
            "data": [
                {
                    "synonyms": [
                        "big apple"
                    ],
                    "value": "new york"
                },
                {
                    "synonyms": [
                        "city of lights"
                    ],
                    "value": "paris"
                },
                {
                    "synonyms": [],
                    "value": "london"
                }
            ],
            "use_synonyms": True,
            "matching_strictness": 1.0
        }
        self.assertDictEqual(expected_entity_dict, entity_dict)

    @patch("pathlib.io")
    @fail_if_not_removed
    def test_from_file_with_autoextensible(self, mock_io):
        # Given
        entity_file = "entity_location.txt"
        location_txt = """# automatically_extensible=false
new york,big apple
paris,city of lights
london
        """

        # pylint:disable=unused-argument
        def mock_open(self_, *args, **kwargs):
            if str(self_) == entity_file:
                return io.StringIO(location_txt)
            return None

        # pylint:enable=unused-argument

        mock_io.open.side_effect = mock_open

        # When
        entity_dataset = Entity.from_file(entity_file)
        entity_dict = entity_dataset.json

        # Then
        expected_entity_dict = {
            "automatically_extensible": False,
            "data": [
                {
                    "synonyms": [
                        "big apple"
                    ],
                    "value": "new york"
                },
                {
                    "synonyms": [
                        "city of lights"
                    ],
                    "value": "paris"
                },
                {
                    "synonyms": [],
                    "value": "london"
                }
            ],
            "use_synonyms": True,
            "matching_strictness": 1.0
        }
        self.assertDictEqual(expected_entity_dict, entity_dict)

    @fail_if_not_removed
    def test_should_fail_generating_entity_with_wrong_file_name(self):
        # Given
        entity_file = "location.txt"

        # When / Then
        with self.assertRaises(EntityFormatError):
            Entity.from_file(entity_file)
