from __future__ import unicode_literals

import io
from copy import deepcopy
from unittest import TestCase

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
    "language": "en",
    "intent_filters": {
        "hello": ["hello world", "hello you"],
        "bye": ["bye"],
    }
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

        filters_yaml = io.StringIO("""
# Intent filters
---
type: intent_filters
hello:
  - hello world
  - hello you
bye:
  - buy
        """)

        dataset_without_filters_files = [
            who_is_game_yaml, get_weather_yaml, location_yaml]
        dataset_files = dataset_without_filters_files + [filters_yaml]

        # When
        dataset_without_filters = Dataset.from_yaml_files(
            "en", dataset_without_filters_files).json
        dataset = Dataset.from_yaml_files(
            "en", dataset_files).json

        # Then
        expected_dataset = deepcopy(EXPECTED_DATASET_DICT)
        validate_and_format_dataset(dataset)
        self.assertDictEqual(expected_dataset, dataset)

        expected_dataset.pop("intent_filters")
        validate_and_format_dataset(dataset_without_filters)
        self.assertDictEqual(expected_dataset, dataset_without_filters)


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

# Intent filters part 1
---
type: intent_filters
hello:
  - hello world
  - hello you
  
# Intent filters part 2
---
type: intent_filters
bye:
  - buy
        """)

        # When
        dataset = Dataset.from_yaml_files("en", [dataset_stream])
        dataset_dict = dataset.json

        # Then
        validate_and_format_dataset(dataset_dict)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset_dict)
