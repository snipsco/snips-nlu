from __future__ import unicode_literals

import io
from unittest import TestCase

from deprecation import fail_if_not_removed
from mock import patch

from snips_nlu.dataset import Intent, IntentFormatError


class TestIntentLoading(TestCase):
    def test_should_load_from_yaml_file(self):
        # Given
        intent = Intent.from_yaml(io.StringIO("""
# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - what is the weather in [weatherLocation:location](paris) ?
  - "Will it rain [date:snips/datetime](tomorrow) in
    [weatherLocation:location](london)?"
        """))

        # When
        intent_dict = intent.json

        # Then
        expected_intent_dict = {
            "utterances": [
                {
                    "data": [
                        {
                            "text": "what is the weather in "
                        },
                        {
                            "text": "paris",
                            "entity": "location",
                            "slot_name": "weatherLocation"
                        },
                        {
                            "text": " ?"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "Will it rain "
                        },
                        {
                            "text": "tomorrow",
                            "entity": "snips/datetime",
                            "slot_name": "date"
                        },
                        {
                            "text": " in "
                        },
                        {
                            "text": "london",
                            "entity": "location",
                            "slot_name": "weatherLocation"
                        },
                        {
                            "text": "?"
                        }
                    ]
                }
            ]
        }
        self.assertDictEqual(expected_intent_dict, intent_dict)

    def test_should_load_from_yaml_file_using_slot_mapping(self):
        # Given
        intent = Intent.from_yaml(io.StringIO("""
# getWeather Intent
---
type: intent
name: getWeather
slots:
  - name: date
    entity: snips/datetime
  - name: weatherLocation
    entity: location
utterances:
  - what is the weather in [weatherLocation](paris) ?
  - Will it rain [date] in [weatherLocation](london)?
        """))

        # When
        intent_dict = intent.json

        # Then
        expected_intent_dict = {
            "utterances": [
                {
                    "data": [
                        {
                            "text": "what is the weather in "
                        },
                        {
                            "text": "paris",
                            "entity": "location",
                            "slot_name": "weatherLocation"
                        },
                        {
                            "text": " ?"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "Will it rain "
                        },
                        {
                            "text": None,
                            "entity": "snips/datetime",
                            "slot_name": "date"
                        },
                        {
                            "text": " in "
                        },
                        {
                            "text": "london",
                            "entity": "location",
                            "slot_name": "weatherLocation"
                        },
                        {
                            "text": "?"
                        }
                    ]
                }
            ]
        }
        self.assertDictEqual(expected_intent_dict, intent_dict)

    def test_should_load_from_yaml_file_using_implicit_values(self):
        # Given
        intent = Intent.from_yaml(io.StringIO("""
# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - what is the weather in [location] ?
        """))

        # When
        intent_dict = intent.json

        # Then
        expected_intent_dict = {
            "utterances": [
                {
                    "data": [
                        {
                            "text": "what is the weather in "
                        },
                        {
                            "text": None,
                            "entity": "location",
                            "slot_name": "location"
                        },
                        {
                            "text": " ?"
                        }
                    ]
                }
            ]
        }
        self.assertDictEqual(expected_intent_dict, intent_dict)

    @patch("pathlib.io")
    @fail_if_not_removed
    def test_should_generate_intent_from_text_file(self, mock_io):
        # Given
        intent_file = "intent_getWeather.txt"
        get_weather_txt = """
what is the weather in [weatherLocation:location](Paris)?
Will it rain [weatherDate:snips/datetime](tomorrow) in [weatherLocation](Moscow)?
How is the weather in [weatherLocation:location] [weatherDate] please?
is it raining in [weatherLocation] [weatherDate:snips/datetime]
        """

        # pylint:disable=unused-argument
        def mock_open(self_, *args, **kwargs):
            if str(self_) == intent_file:
                return io.StringIO(get_weather_txt)
            return None

        # pylint:enable=unused-argument

        mock_io.open.side_effect = mock_open

        # When
        intent_dataset = Intent.from_file(intent_file)
        intent_dict = intent_dataset.json

        # Then
        expected_intent_dict = {
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
                            "text": "Will it rain "
                        },
                        {
                            "entity": "snips/datetime",
                            "slot_name": "weatherDate",
                            "text": "tomorrow"
                        },
                        {
                            "text": " in "
                        },
                        {
                            "entity": "location",
                            "slot_name": "weatherLocation",
                            "text": "Moscow"
                        },
                        {
                            "text": "?"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "How is the weather in "
                        },
                        {
                            "entity": "location",
                            "slot_name": "weatherLocation",
                            "text": None
                        },
                        {
                            "entity": "snips/datetime",
                            "slot_name": "weatherDate",
                            "text": None
                        },
                        {
                            "text": " please?"
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
                            "text": None
                        },
                        {
                            "entity": "snips/datetime",
                            "slot_name": "weatherDate",
                            "text": None
                        }
                    ]
                }
            ]
        }

        self.assertDictEqual(expected_intent_dict, intent_dict)

    @fail_if_not_removed
    def test_should_fail_generating_intent_with_wrong_file_name(self):
        # Given
        intent_file = "getWeather.txt"

        # When / Then
        with self.assertRaises(IntentFormatError):
            Intent.from_file(intent_file)
