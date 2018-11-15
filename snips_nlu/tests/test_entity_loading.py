import io
from unittest import TestCase

import yaml

from snips_nlu.constants import PACKAGE_PATH
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

    def test_from_text_file(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        entity_file = examples_path / "entity_location.txt"

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

    def test_from_file_with_autoextensible(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        entity_file = examples_path / "entity_location_autoextent_false.txt"

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

    def test_should_fail_generating_entity_with_wrong_file_name(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        entity_file = examples_path / "location.txt"

        # When / Then
        with self.assertRaises(EntityFormatError):
            Entity.from_file(entity_file)
