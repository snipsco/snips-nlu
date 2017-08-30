# coding=utf-8
from __future__ import unicode_literals

import unittest

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.constants import CUSTOM_ENGINE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.tests.utils import SAMPLE_DATASET


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
            "snips_nlu_version": "1.1.1"
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
            "language": "en",
            "snips_nlu_version": "1.1.1"
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
            "language": "en",
            "snips_nlu_version": "1.1.1"
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
            "language": "en",
            "snips_nlu_version": "1.1.1"
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
            "language": "eng",
            "snips_nlu_version": "1.1.1"
        }

        # When/Then
        with self.assertRaises(ValueError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(ctx.exception.message,
                         "Language name must be ISO 639-1, found 'eng'")

    def test_should_format_dataset_by_adding_synonyms(self):
        # Given
        dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "Entity_1",
                            "synonyms": ["entity 2"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                }
            },
            "language": "en",
            "snips_nlu_version": "1.1.1"
        }

        expected_dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "Entity_1",
                            "synonyms": ["entity 2", "entity_1"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "capitalize": True
                }
            },
            "language": "en",
            "snips_nlu_version": "1.1.1"
        }
        capitalization_threshold = .1

        # When
        dataset = validate_and_format_dataset(
            dataset, capitalization_threshold=capitalization_threshold)

        # Then
        self.assertDictEqual(dataset, expected_dataset)

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
                },
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
            "language": "en",
            "snips_nlu_version": "1.1.1"
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
                    "automatically_extensible": False,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "1.1.1"
        }
        capitalization_threshold = .1

        # When
        dataset = validate_and_format_dataset(
            dataset, capitalization_threshold=capitalization_threshold)

        # Then
        self.assertEqual(dataset, expected_dataset)

    def test_dataset_should_require_nlu_version(self):
        # Given
        dataset = dict(SAMPLE_DATASET)

        # When
        dataset.pop("snips_nlu_version")

        # Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        expected_exception_message = "Expected dataset to have key:" \
                                     " 'snips_nlu_version'"
        self.assertEqual(ctx.exception.message, expected_exception_message)

    def test_dataset_require_semantic_version(self):
        # Given
        dataset = dict(SAMPLE_DATASET)

        # When
        dataset["snips_nlu_version"] = "1.1.1.1.1.1.1.1.1.1"

        # Then
        with self.assertRaises(ValueError) as ctx:
            validate_and_format_dataset(dataset)
        expected_exception_message = "Invalid version string:" \
                                     " u'1.1.1.1.1.1.1.1.1.1'"
        self.assertEqual(ctx.exception.message, expected_exception_message)

    def test_should_add_missing_reference_entity_values_when_not_use_synonyms(
            self):
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
                            "synonyms": ["entity 1", "alternative entity 1"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
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
                            "synonyms": ["entity 1", "alternative entity 1"]
                        },
                        {
                            "value": "alternative entity 1",
                            "synonyms": ["alternative entity 1"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": False,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1",
        }
        capitalization_threshold = .1

        # When
        dataset = validate_and_format_dataset(
            dataset, capitalization_threshold)

        # Then
        self.maxDiff = None
        self.assertEqual(dataset, expected_dataset)

    def test_should_not_require_data_for_builtin_entities(self):
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
                                    "text": "10p.m",
                                    "entity": BuiltInEntity.DATETIME.label,
                                    "slot_name": "startTime"
                                }
                            ]
                        }
                    ],
                    "engineType": CUSTOM_ENGINE
                }
            },
            "entities": {
                BuiltInEntity.DATETIME.label: {}
            },
            "language": "en",
            "snips_nlu_version": "0.1.0"
        }

        # When / Then
        try:
            validate_and_format_dataset(dataset)
        except:
            self.fail("Could not validate dataset")

    def test_should_remove_empty_entities_value_and_empty_synonyms(self):
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
                                    "text": "",
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
                            "synonyms": [""]
                        },
                        {
                            "value": "",
                            "synonyms": []
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
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
                                    "text": "",
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
                            "synonyms": ["entity 1"]
                        }
                    ],
                    "capitalize": False,
                    "use_synonyms": False,
                    "automatically_extensible": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }
        capitalization_threshold = .1

        # When
        dataset = validate_and_format_dataset(
            dataset, capitalization_threshold)

        # Then
        self.assertEqual(dataset, expected_dataset)

    def test_should_add_capitalize_field(self):
        # Given
        capitalization_threshold = .3
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "My entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot0"
                                },
                                {
                                    "text": "entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot2"
                                },
                                {
                                    "text": "entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot2"
                                },
                                {
                                    "text": "entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot3"
                                },
                                {
                                    "text": "My entity2",
                                    "entity": "entity2",
                                    "slot_name": "slot1"
                                },
                                {
                                    "text": "myentity2",
                                    "entity": "entity2",
                                    "slot_name": "slot1"
                                },
                                {
                                    "text": "m_entity3",
                                    "entity": "entity3",
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
                    "data": [],
                    "use_synonyms": False,
                    "automatically_extensible": True
                },
                "entity2": {
                    "data": [],
                    "use_synonyms": False,
                    "automatically_extensible": True
                },
                "entity3": {
                    "data": [
                        {
                            "value": "Entity3",
                            "synonyms": ["entity3"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": True
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }

        expected_dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "My entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot0"
                                },
                                {
                                    "text": "entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot2"
                                },
                                {
                                    "text": "entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot2"
                                },
                                {
                                    "text": "entity1",
                                    "entity": "entity1",
                                    "slot_name": "slot3"
                                },
                                {
                                    "text": "My entity2",
                                    "entity": "entity2",
                                    "slot_name": "slot1"
                                },
                                {
                                    "text": "myentity2",
                                    "entity": "entity2",
                                    "slot_name": "slot1"
                                },
                                {
                                    "text": "m_entity3",
                                    "entity": "entity3",
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
                            "value": "My entity1",
                            "synonyms": ["my entity1"]
                        },
                        {
                            "value": "entity1",
                            "synonyms": ["entity1"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "capitalize": False
                },
                "entity2": {
                    "data": [
                        {
                            "value": "My entity2",
                            "synonyms": ["my entity2"]

                        },
                        {
                            "value": "myentity2",
                            "synonyms": ["myentity2"]

                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "capitalize": True
                },
                "entity3": {
                    "data": [
                        {
                            "value": "Entity3",
                            "synonyms": ["entity3"]
                        },
                        {
                            "value": "m_entity3",
                            "synonyms": ["m_entity3"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "capitalize": True
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }

        # When
        dataset = validate_and_format_dataset(
            dataset, capitalization_threshold)

        # Then
        self.assertDictEqual(dataset, expected_dataset)

    def test_should_normalize_synonyms(self):
        # Given
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "éNtity",
                                    "entity": "entity1",
                                    "slot_name": "startTime"
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
                    "automatically_extensible": True
                }
            },
            "language": "en",
            "snips_nlu_version": "0.1.0"
        }

        expected_dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "éNtity",
                                    "entity": "entity1",
                                    "slot_name": "startTime"
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
                            "value": "éNtity",
                            "synonyms": ["entity"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": True,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.1.0"
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertDictEqual(dataset, expected_dataset)


if __name__ == '__main__':
    unittest.main()
