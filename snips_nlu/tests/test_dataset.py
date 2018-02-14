# coding=utf-8
from __future__ import unicode_literals

import unittest
from builtins import str

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.constants import (
    ENTITIES, AUTOMATICALLY_EXTENSIBLE, UTTERANCES, CAPITALIZE)
from snips_nlu.dataset import validate_and_format_dataset


class TestDataset(unittest.TestCase):
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
            "entities": {},
            "language": "en",
            "snips_nlu_version": "1.1.1"
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(str(ctx.exception.args[0]),
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
            },
            "language": "en",
            "snips_nlu_version": "1.1.1"
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(str(ctx.exception.args[0]),
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
        self.assertEqual(str(ctx.exception.args[0]),
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
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual(str(ctx.exception.args[0]), "Unknown iso_code 'eng'")

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
                    "utterances": {
                        "Entity_1": "Entity_1",
                        "Entity_one": "Entity_1",
                        "Entity1": "Entity_1",
                        "entity_1": "Entity_1",
                        "entity_one": "Entity_1",
                        "entity1": "Entity_1",
                        "entity 2": "Entity_1",
                        "entity two": "Entity_1",
                    },
                    "automatically_extensible": False,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "1.1.1",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

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
                    ]
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
                    ]
                }
            },
            "entities": {
                "entity1": {
                    "utterances": {
                        "entity 1 bis": "entity 1",
                        "entity one bis": "entity 1",
                        "entity 1": "entity 1",
                        "entity one": "entity 1",
                        "alternative entity 1": "alternative entity 1",
                        "alternative entity one": "alternative entity 1"
                    },
                    "automatically_extensible": False,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "1.1.1",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(dataset, expected_dataset)

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
                    ]
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
                    ]
                }
            },
            "entities": {
                "entity1": {
                    "utterances":
                        {
                            "alternative entity 1": "alternative entity 1",
                            "alternative entity one": "alternative entity 1",
                            "entity 1": "entity 1",
                            "entity one": "entity 1",
                        },
                    "automatically_extensible": False,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
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
                    ]
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
        except:  # pylint: disable=W0702
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
                    ]
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
                    ]
                }
            },
            "entities": {
                "entity1": {
                    "utterances":
                        {
                            "entity 1": "entity 1",
                            "entity one": "entity 1"
                        },
                    "capitalize": False,
                    "automatically_extensible": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(dataset, expected_dataset)

    def test_should_add_capitalize_field(self):
        # Given
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
                    ]
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
                    ]
                }
            },
            "entities": {
                "entity1": {
                    "utterances":
                        {
                            "My entity1": "My entity1",
                            "my entity1": "My entity1",
                            "entity1": "entity1"
                        },
                    "automatically_extensible": True,
                    "capitalize": True
                },
                "entity2": {
                    "utterances": {
                        "My entity2": "My entity2",
                        "my entity2": "My entity2",
                        "myentity2": "myentity2"
                    },
                    "automatically_extensible": True,
                    "capitalize": True
                },
                "entity3": {
                    "utterances":
                        {
                            "Entity3": "Entity3",
                            "entity3": "Entity3",
                            "m_entity3": "m_entity3",
                            "mentity3": "m_entity3"
                        },
                    "automatically_extensible": True,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

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
                    ]
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
                    ]
                }
            },
            "entities": {
                "entity1": {
                    "utterances": {
                        "éNtity": "éNtity",
                        "entity": "éNtity",
                    },
                    "automatically_extensible": True,
                    "capitalize": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.1.0",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertDictEqual(dataset, expected_dataset)

    def test_dataset_should_handle_synonyms(self):
        # Given
        dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "Ëntity 1",
                            "synonyms": ["entity 2"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": True
                }
            },
            "language": "en",
            "snips_nlu_version": "1.1.1"
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        expected_entities = {
            "entity1": {
                AUTOMATICALLY_EXTENSIBLE: True,
                UTTERANCES: {
                    "Ëntity 1": "Ëntity 1",
                    "Ëntity one": "Ëntity 1",
                    "entity 1": "Ëntity 1",
                    "entity one": "Ëntity 1",
                    "entity 2": "Ëntity 1",
                    "entity two": "Ëntity 1"
                },
                CAPITALIZE: False
            }
        }

        # Then
        self.assertDictEqual(dataset[ENTITIES], expected_entities)
