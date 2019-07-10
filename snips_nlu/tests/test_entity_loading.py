from __future__ import unicode_literals

import io
from unittest import TestCase

from snips_nlu.dataset import Entity
from snips_nlu.exceptions import EntityFormatError


class TestEntityLoading(TestCase):
    def test_from_yaml_file(self):
        # Given
        entity = Entity.from_yaml(io.StringIO("""
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
        """))

        # When
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
        entity = Entity.from_yaml(io.StringIO("""
# Location Entity
---
name: location
values:
- [new york, big apple]
- [paris, city of lights]
- london
        """))

        # When
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

        # When / Then
        with self.assertRaises(EntityFormatError):
            Entity.from_yaml(yaml_stream)

    def test_fail_from_yaml_file_when_no_name(self):
        # Given
        entity_io = io.StringIO("""
# Location Entity
---
values:
- [new york, big apple]
- [paris, city of lights]
- london
        """)

        # When / Then
        with self.assertRaises(EntityFormatError):
            Entity.from_yaml(entity_io)
