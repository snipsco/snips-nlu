# coding=utf-8
from __future__ import unicode_literals

from builtins import str

from mock import mock

from snips_nlu.constants import (
    ENTITIES, SNIPS_DATETIME)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.tests.utils import SnipsTest


class TestDataset(SnipsTest):
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
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Expected chunk to have key: 'slot_name'",
                         str(ctx.exception.args[0]))

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
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Expected entities to have key: 'unknown_entity'",
                         str(ctx.exception.args[0]))

    def test_missing_entity_key_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [],
                    "automatically_extensible": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
        }

        # When/Then
        with self.assertRaises(KeyError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Expected entity to have key: 'use_synonyms'",
                         str(ctx.exception.args[0]))

    def test_missing_parser_threshold_should_be_handled(self):
        # TODO: This test is temporary, and must be removed once the backward
        # compatibility with the previous dataset format (without
        # "parser_threshold"), gets deprecated.

        # Given
        dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "data": [],
                    "automatically_extensible": False,
                    "use_synonyms": True
                }
            },
            "language": "en",
        }

        # When/Then
        dataset = validate_and_format_dataset(dataset)

        self.assertEqual(
            1.0, dataset["entities"]["entity1"].get("parser_threshold"))

    def test_invalid_language_should_raise_exception(self):
        # Given
        dataset = {
            "intents": {},
            "entities": {},
            "language": "eng",
        }

        # When/Then
        with self.assertRaises(ValueError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Unknown language: 'eng'", str(ctx.exception.args[0]))

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_should_format_dataset_by_adding_synonyms(
            self, mocked_get_string_variations):
        # Given
        # pylint: disable=unused-argument
        def mock_get_string_variations(variation, language,
                                       builtin_entity_parser):
            return {variation.lower(), variation.title()}

        mocked_get_string_variations.side_effect = mock_get_string_variations
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
                    "automatically_extensible": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
        }

        expected_dataset = {
            "intents": {},
            "entities": {
                "entity1": {
                    "utterances": {
                        "Entity_1": "Entity_1",
                        "entity_1": "Entity_1",
                        "entity 2": "Entity_1",
                        "Entity 2": "Entity_1",
                    },
                    "automatically_extensible": False,
                    "capitalize": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertDictEqual(expected_dataset, dataset)

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_should_format_dataset_by_adding_entity_values(
            self, mocked_get_string_variations):
        # Given
        # pylint: disable=unused-argument
        def mock_get_string_variations(variation, language,
                                       builtin_entity_parser):
            return {variation, variation.title()}

        mocked_get_string_variations.side_effect = mock_get_string_variations
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
                    "automatically_extensible": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
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
                        "Entity 1 Bis": "entity 1",
                        "entity 1": "entity 1",
                        "Entity 1": "entity 1",
                        "alternative entity 1": "alternative entity 1",
                        "Alternative Entity 1": "alternative entity 1",
                    },
                    "automatically_extensible": False,
                    "capitalize": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(expected_dataset, dataset)

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_should_add_missing_reference_entity_values_when_not_use_synonyms(
            self, mocked_get_string_variations):
        # Given
        # pylint: disable=unused-argument
        def mock_get_string_variations(variation, language,
                                       builtin_entity_parser):
            return {variation}

        mocked_get_string_variations.side_effect = mock_get_string_variations
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
                    "automatically_extensible": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
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
                            "entity 1": "entity 1",
                        },
                    "automatically_extensible": False,
                    "capitalize": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(expected_dataset, dataset)

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
                                    "entity": SNIPS_DATETIME,
                                    "slot_name": "startTime"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                SNIPS_DATETIME: {}
            },
            "language": "en",
        }

        # When / Then
        with self.fail_if_exception("Could not validate dataset"):
            validate_and_format_dataset(dataset)

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_should_remove_empty_entities_value_and_empty_synonyms(
            self, mocked_get_string_variations):
        # Given
        # pylint: disable=unused-argument
        def mock_get_string_variations(variation, language,
                                       builtin_entity_parser):
            return {variation, variation.title()}

        mocked_get_string_variations.side_effect = mock_get_string_variations
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
                    "automatically_extensible": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
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
                            "Entity 1": "entity 1",
                        },
                    "capitalize": False,
                    "automatically_extensible": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertEqual(expected_dataset, dataset)

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_should_add_capitalize_field(
            self, mocked_get_string_variations):
        # Given
        # pylint: disable=unused-argument
        def mock_get_string_variations(variation, language,
                                       builtin_entity_parser):
            return {variation, variation.title()}

        mocked_get_string_variations.side_effect = mock_get_string_variations
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
                    "automatically_extensible": True,
                    "parser_threshold": 1.0
                },
                "entity2": {
                    "data": [],
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "parser_threshold": 1.0
                },
                "entity3": {
                    "data": [
                        {
                            "value": "Entity3",
                            "synonyms": ["entity3"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
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
                            "My Entity1": "My entity1",
                            "entity1": "entity1",
                            "Entity1": "entity1",
                        },
                    "automatically_extensible": True,
                    "capitalize": True,
                    "parser_threshold": 1.0
                },
                "entity2": {
                    "utterances": {
                        "My entity2": "My entity2",
                        "My Entity2": "My entity2",
                        "myentity2": "myentity2",
                        "Myentity2": "myentity2"
                    },
                    "automatically_extensible": True,
                    "capitalize": True,
                    "parser_threshold": 1.0
                },
                "entity3": {
                    "utterances":
                        {
                            "Entity3": "Entity3",
                            "m_entity3": "m_entity3",
                            "M_Entity3": "m_entity3"
                        },
                    "automatically_extensible": True,
                    "capitalize": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertDictEqual(expected_dataset, dataset)

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_should_normalize_synonyms(
            self, mocked_get_string_variations):
        # Given
        # pylint: disable=unused-argument
        def mock_get_string_variations(variation, language,
                                       builtin_entity_parser):
            return {variation.lower(), variation.title()}

        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "ëNtity",
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
                    "automatically_extensible": True,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
        }

        expected_dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "ëNtity",
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
                        "ëntity": "ëNtity",
                        "Ëntity": "ëNtity",
                        "ëNtity": "ëNtity"
                    },
                    "automatically_extensible": True,
                    "capitalize": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
            "validated": True
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        self.assertDictEqual(expected_dataset, dataset)

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_dataset_should_handle_synonyms(
            self, mocked_get_string_variations):
        # Given
        # pylint: disable=unused-argument
        def mock_get_string_variations(variation, language,
                                       builtin_entity_parser):
            return {variation.lower(), variation.title()}

        mocked_get_string_variations.side_effect = mock_get_string_variations
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
                    "automatically_extensible": True,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        expected_entities = {
            "entity1": {
                "automatically_extensible": True,
                "utterances": {
                    "Ëntity 1": "Ëntity 1",
                    "ëntity 1": "Ëntity 1",
                    "entity 2": "Ëntity 1",
                    "Entity 2": "Ëntity 1",
                },
                "capitalize": False,
                "parser_threshold": 1.0
            }
        }

        # Then
        self.assertDictEqual(dataset[ENTITIES], expected_entities)

    def test_should_not_avoid_synomyms_variations_collision(self):
        # Given
        dataset = {
            "intents": {
                "dummy_but_tricky_intent": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy_value",
                                    "entity": "dummy_but_tricky_entity",
                                    "slot_name": "dummy_but_tricky_slot"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "dummy_but_tricky_entity": {
                    "data": [
                        {
                            "value": "a",
                            "synonyms": [
                                "favorïte"
                            ]
                        },
                        {
                            "value": "b",
                            "synonyms": [
                                "favorite"
                            ]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "parser_threshold": 1.0
                }
            },
            "language": "en",
        }

        # When
        dataset = validate_and_format_dataset(dataset)

        # Then
        entity = dataset["entities"]["dummy_but_tricky_entity"]
        expected_utterances = {
            "A": "a",
            "B": "b",
            "DummyValue": "dummy_value",
            "Dummy_Value": "dummy_value",
            "Favorïte": "a",
            "a": "a",
            "b": "b",
            "dummy_value": "dummy_value",
            "dummyvalue": "dummy_value",
            "favorite": "b",
            "favorïte": "a"
        }
        self.assertDictEqual(expected_utterances, entity["utterances"])
