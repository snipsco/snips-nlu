from unittest import TestCase

from snips_nlu.constants import PACKAGE_PATH
from snips_nlu.dataset import Intent, IntentFormatError


class TestIntentLoading(TestCase):
    def test_should_generate_intent_from_text_file(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        intent_file = examples_path / "intent_getWeather.txt"

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
                            "entity": None,
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
                            "text": "San Francisco"
                        },
                        {
                            "entity": None,
                            "slot_name": "weatherDate",
                            "text": None
                        },
                        {
                            "text": " please?"
                        }
                    ]
                }
            ]
        }

        self.assertDictEqual(expected_intent_dict, intent_dict)

    def test_should_fail_generating_intent_with_wrong_file_name(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        intent_file = examples_path / "getWeather.txt"

        # When / Then
        with self.assertRaises(IntentFormatError):
            Intent.from_file(intent_file)
