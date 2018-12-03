from __future__ import unicode_literals

import io
from unittest import TestCase

from deprecation import fail_if_not_removed
from mock import patch

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
    @patch("snips_nlu.dataset.dataset.io")
    def test_should_generate_dataset_from_yaml_files(self, mock_io):
        # Given
        intent_file_1 = "whoIsGame.yaml"
        intent_file_2 = "getWeather.yaml"
        entity_file_1 = "location.yaml"

        who_is_game_yaml = """
# whoIsGame Intent
---
type: intent
name: whoIsGame
utterances:
  - who is the [role](president) of [country](France)
  - who is the [role](CEO) of [company](Google) please
        """

        get_weather_yaml = """
# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - what is the weather in [weatherLocation:location](Paris)?
  - is it raining in [weatherLocation] [weatherDate:snips/datetime]
        """

        location_yaml = """
# Location Entity
---
type: entity
name: location
automatically_extensible: true
values:
- [new york, big apple]
- london
        """

        # pylint:disable=unused-argument
        def mock_open(filename, **kwargs):
            if filename == intent_file_1:
                return io.StringIO(who_is_game_yaml)
            if filename == intent_file_2:
                return io.StringIO(get_weather_yaml)
            if filename == entity_file_1:
                return io.StringIO(location_yaml)
            return None

        # pylint:enable=unused-argument

        mock_io.open.side_effect = mock_open
        dataset_files = [intent_file_1, intent_file_2, entity_file_1]

        # When
        dataset = Dataset.from_yaml_files("en", dataset_files)
        dataset_dict = dataset.json

        # Then
        validate_and_format_dataset(dataset_dict)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset_dict)

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
        dataset = Dataset.from_yaml_files("en", [dataset_stream])
        dataset_dict = dataset.json

        # Then
        validate_and_format_dataset(dataset_dict)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset_dict)

    @fail_if_not_removed
    def test_should_generate_dataset_from_files(self):
        # Given
        intent_file_1 = "intent_whoIsGame.txt"
        intent_file_2 = "intent_getWeather.txt"
        entity_file_1 = "entity_location.txt"

        who_is_game_txt = """
who is the [role:role](president) of [country:country](France)
who is the [role:role](CEO) of [company:company](Google) please
"""

        get_weather_txt = """
what is the weather in [weatherLocation:location](Paris)?
is it raining in [weatherLocation] [weatherDate:snips/datetime]
"""

        location_txt = """
new york,big apple
london
        """

        # pylint:disable=unused-argument
        def mock_open(self_, *args, **kwargs):
            if str(self_) == intent_file_1:
                return io.StringIO(who_is_game_txt)
            if str(self_) == intent_file_2:
                return io.StringIO(get_weather_txt)
            if str(self_) == entity_file_1:
                return io.StringIO(location_txt)
            return None

        # pylint:enable=unused-argument

        dataset_files = [intent_file_1, intent_file_2, entity_file_1]

        # When
        with patch("pathlib.io") as mock_io:
            mock_io.open.side_effect = mock_open
            dataset = Dataset.from_files("en", dataset_files)
        dataset_dict = dataset.json

        # When / Then
        validate_and_format_dataset(dataset_dict)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset_dict)
