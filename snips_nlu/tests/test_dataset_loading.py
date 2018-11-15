from unittest import TestCase

from snips_nlu.constants import PACKAGE_PATH
from snips_nlu.dataset import AssistantDataset, validate_and_format_dataset


class TestDatasetLoading(TestCase):
    def test_should_generate_dataset_from_files(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        intent_file_1 = examples_path / "intent_whoIsGame.txt"
        intent_file_2 = examples_path / "intent_getWeather.txt"
        entity_file_1 = examples_path / "entity_location.txt"

        dataset = AssistantDataset.from_files(
            "en", [intent_file_1, intent_file_2, entity_file_1])
        dataset_dict = dataset.json

        # When / Then
        expected_dataset_dict = {
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
                                    "text": "San Francisco"
                                },
                                {
                                    "entity": "snips/datetime",
                                    "slot_name": "weatherDate",
                                    "text": "today"
                                },
                                {
                                    "text": "?"
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
                                    "text": "prime minister"
                                },
                                {
                                    "text": " of "
                                },
                                {
                                    "entity": "country",
                                    "slot_name": "country",
                                    "text": "UK"
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
        validate_and_format_dataset(dataset_dict)
        self.assertDictEqual(expected_dataset_dict, dataset_dict)
