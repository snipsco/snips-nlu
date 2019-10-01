from __future__ import unicode_literals

import io
from unittest import TestCase

import mock

from snips_nlu.dataset import Dataset, validate_and_format_dataset

EXPECTED_DATASET_DICT = {
    "entities": {
        "company": {
            "automatically_extensible": True,
            "data": [],
            "use_synonyms": True,
            "matching_strictness": 1.0,
        },
        "country": {
            "automatically_extensible": True,
            "data": [],
            "use_synonyms": True,
            "matching_strictness": 1.0,
        },
        "location": {
            "automatically_extensible": True,
            "data": [
                {
                    "synonyms": [
                        "big apple"
                    ],
                    "value": "new york"
                },
                {
                    "synonyms": [],
                    "value": "london"
                }
            ],
            "use_synonyms": True,
            "matching_strictness": 1.0,
        },
        "role": {
            "automatically_extensible": True,
            "data": [],
            "use_synonyms": True,
            "matching_strictness": 1.0,
        },
        "snips/datetime": {}
    },
    "intents": {
        "getWeather": {
            "utterances": [
                {
                    "data": [
                        {
                            "text": "what is the weather in "
                        },
                        {
                            "entity": "location",
                            "slot_name": "weatherLocation",
                            "text": "Paris"
                        },
                        {
                            "text": "?"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "is it raining in "
                        },
                        {
                            "entity": "location",
                            "slot_name": "weatherLocation",
                            "text": "new york"
                        },
                        {
                            "text": " "
                        },
                        {
                            "entity": "snips/datetime",
                            "slot_name": "weatherDate",
                            "text": "Today"
                        }
                    ]
                }
            ]
        },
        "whoIsGame": {
            "utterances": [
                {
                    "data": [
                        {
                            "text": "who is the "
                        },
                        {
                            "entity": "role",
                            "slot_name": "role",
                            "text": "president"
                        },
                        {
                            "text": " of "
                        },
                        {
                            "entity": "country",
                            "slot_name": "country",
                            "text": "France"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "who is the "
                        },
                        {
                            "entity": "role",
                            "slot_name": "role",
                            "text": "CEO"
                        },
                        {
                            "text": " of "
                        },
                        {
                            "entity": "company",
                            "slot_name": "company",
                            "text": "Google"
                        },
                        {
                            "text": " please"
                        }
                    ]
                }
            ]
        }
    },
    "language": "en"
}


class TestDatasetLoading(TestCase):
    def test_should_generate_dataset_from_yaml_files(self):
        # Given
        who_is_game_yaml = io.StringIO("""
# whoIsGame Intent
---
type: intent
name: whoIsGame
utterances:
  - who is the [role](president) of [country](France)
  - who is the [role](CEO) of [company](Google) please
        """)

        get_weather_yaml = io.StringIO("""
# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - what is the weather in [weatherLocation:location](Paris)?
  - is it raining in [weatherLocation] [weatherDate:snips/datetime]
        """)

        location_yaml = io.StringIO("""
# Location Entity
---
type: entity
name: location
automatically_extensible: true
values:
- [new york, big apple]
- london
        """)

        dataset_files = [who_is_game_yaml, get_weather_yaml, location_yaml]

        # When
        with mock.patch("snips_nlu_parsers.get_builtin_entity_examples",
                        return_value=["Today"]):
            dataset = Dataset.from_yaml_files("en", dataset_files)

        # Then
        validate_and_format_dataset(dataset)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset.json)

    def test_should_generate_dataset_from_merged_yaml_file(self):
        # Given
        dataset_stream = io.StringIO("""
# whoIsGame Intent
---
type: intent
name: whoIsGame
utterances:
  - who is the [role](president) of [country](France)
  - who is the [role](CEO) of [company](Google) please

# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - what is the weather in [weatherLocation:location](Paris)?
  - is it raining in [weatherLocation] [weatherDate:snips/datetime]
  
# Location Entity
---
type: entity
name: location
automatically_extensible: true
values:
- [new york, big apple]
- london
        """)

        # When
        with mock.patch("snips_nlu_parsers.get_builtin_entity_examples",
                        return_value=["Today"]):
            dataset = Dataset.from_yaml_files("en", [dataset_stream])

        # Then
        validate_and_format_dataset(dataset)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset.json)
