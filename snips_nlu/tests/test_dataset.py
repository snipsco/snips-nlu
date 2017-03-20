import unittest

from snips_nlu.dataset import validate_dataset


class TestDataset(unittest.TestCase):
    def test_invalid_intent_name_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {
                "invalid/intent_name": {
                    "utterances": [],
                }
            },
            "entities": {}
        }

        # When/Then
        with self.assertRaises(AssertionError) as ctx:
            validate_dataset(dataset)
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
                    ]
                }
            },
            "entities": {}
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_dataset(dataset)
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
                    ]
                }
            },
            "entities": {
                "entity1": {
                    "data": [],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            }
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_dataset(dataset)
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
            }
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_dataset(dataset)
        self.assertEqual(ctx.exception.message,
                         "Expected entity to have key: 'use_synonyms'")


if __name__ == '__main__':
    unittest.main()
