from __future__ import unicode_literals

import io
from unittest import TestCase

from snips_nlu.dataset import Intent
from snips_nlu.exceptions import IntentFormatError


class TestIntentLoading(TestCase):
    def test_should_load_from_yaml_file(self):
        # Given
        intent = Intent.from_yaml(io.StringIO("""
# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - "what is the weather in [weatherLocation:location](paris) 
    [date:snips/datetime](today) ?"
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
                            "text": " "
                        },
                        {
                            "text": "today",
                            "entity": "snips/datetime",
                            "slot_name": "date"
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

    def test_should_raise_when_missing_bracket_in_utterance(self):
        # Given
        intent_io = io.StringIO("""
# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - what is the weather in [location] ?
  - give me the weather forecast in [location tomorrow please
  - what's the weather in [location] this weekend ?
        """)

        # When / Then
        with self.assertRaises(IntentFormatError) as cm:
            Intent.from_yaml(intent_io)

        faulty_utterance = "give me the weather forecast in [location " \
                           "tomorrow please"

        self.assertTrue(faulty_utterance in str(cm.exception))

    def test_should_raise_when_missing_parenthesis_in_utterance(self):
        # Given
        intent_io = io.StringIO("""
# getWeather Intent
---
type: intent
name: getWeather
utterances:
  - what is the weather in [location] ?
  - give me the weather forecast in [location] tomorrow please
  - what's the weather in [location](Paris this weekend ?
        """)

        # When / Then
        with self.assertRaises(IntentFormatError) as cm:
            Intent.from_yaml(intent_io)

        faulty_utterance = "what's the weather in [location](Paris this " \
                           "weekend ?"

        self.assertTrue(faulty_utterance in str(cm.exception))
