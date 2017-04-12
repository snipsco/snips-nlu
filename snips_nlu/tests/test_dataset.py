import unittest

from snips_nlu.constants import CUSTOM_ENGINE
from snips_nlu.dataset import validate_and_format_dataset


class TestDataset(unittest.TestCase):
    def test_invalid_intent_name_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {
                "invalid/intent_name": {
                    "utterances": [],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {},
            "language": "en",
        }

        # When/Then
        with self.assertRaises(AssertionError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(ctx.exception.message,
                         "invalid/intent_name is an invalid intent name. "
                         "Intent names must only use: [a-zA-Z0-9_- ]")

    def test_missing_intent_key_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "unknown entity",
                                    "entity": "unknown_entity"
                                }
                            ]
                        }
                    ],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {},
            "language": "en"
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(ctx.exception.message,
                         "Expected chunk to have key: 'slot_name'")

    def test_unknown_entity_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "unknown entity",
                                    "entity": "unknown_entity",
                                    "slot_name": "unknown_entity_slot"
                                }
                            ]
                        }
                    ],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {
                "entity1": {
                    "data": [],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(ctx.exception.message,
                         "Expected entities to have key: 'unknown_entity'")

    def test_missing_entity_key_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [],
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(ctx.exception.message,
                         "Expected entity to have key: 'use_synonyms'")

    def test_invalid_language_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {},
            "entities": {},
            "language": "eng"
        }

        # When/Then
        with self.assertRaises(ValueError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(ctx.exception.message,
                         "Language name must be ISO 639-1, found 'eng'")

    def test_should_format_dataset_by_adding_utterance_text(self):
        # Given
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "entity 1",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ]
                        }
                    ],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": []
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        expected_dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "entity 1",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ],
                            "utterance_text": "this is entity 1"
                        }
                    ],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["entity 1"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(dataset, expected_dataset)

    def test_should_format_dataset_by_adding_synonyms(self):
        # Given
        dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": []
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        expected_dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["entity 1"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(dataset, expected_dataset)

    def test_should_format_dataset_by_adding_entity_values(self):
        # Given
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "alternative entity 1",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "entity 1",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ]
                        }
                    ],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["entity 1", "entity 1 bis"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        expected_dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "alternative entity 1",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ],
                            "utterance_text": "this is alternative entity 1"
                        },
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "entity 1",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ],
                            "utterance_text": "this is entity 1"
                        }
                    ],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["entity 1", "entity 1 bis"]
                        },
                        {
                            "value": "alternative entity 1",
                            "synonyms": ["alternative entity 1"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(dataset, expected_dataset)


if __name__ == '__main__':
    unittest.main()
