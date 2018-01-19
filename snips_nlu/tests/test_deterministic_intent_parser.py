# coding=utf-8
from __future__ import unicode_literals

import re
import unittest
from builtins import range

from mock import patch

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.constants import (RES_MATCH_RANGE, VALUE, ENTITY, DATA, TEXT,
                                 SLOT_NAME, RES_INTENT_NAME)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.deterministic_intent_parser import (
    DeterministicIntentParser, deduplicate_overlapping_slots,
    replace_builtin_entities, preprocess_builtin_entities)
from snips_nlu.languages import Language
from snips_nlu.pipeline.configs.intent_parser import \
    DeterministicIntentParserConfig
from snips_nlu.result import intent_classification_result, _slot
from snips_nlu.tests.utils import SAMPLE_DATASET


class TestDeterministicIntentParser(unittest.TestCase):
    def setUp(self):
        self.intent_dataset = {
            "entities": {
                "dummy_entity_1": {
                    "automatically_extensible": False,
                    "use_synonyms": True,
                    "data": [
                        {
                            "synonyms": [
                                "dummy_a",
                                "dummy 2a",
                                "dummy a",
                                "2 dummy a"
                            ],
                            "value": "dummy_a"
                        },
                        {
                            "synonyms": [
                                "dummy_b",
                                "dummy_bb",
                                "dummy b"
                            ],
                            "value": "dummy_b"
                        },
                        {
                            "synonyms": [
                                "dummy d"
                            ],
                            "value": "dummy d"
                        }
                    ]
                },
                "dummy_entity_2": {
                    "automatically_extensible": False,
                    "use_synonyms": True,
                    "data": [
                        {
                            "synonyms": [
                                "dummy_c",
                                "dummy_cc",
                                "dummy c",
                                "3p.m."
                            ],
                            "value": "dummy_c"
                        }
                    ]
                },
                "snips/datetime": {}
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "This is a first "
                                },
                                {
                                    "text": "dummy_1",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                },
                                {
                                    "text": " query with a second "
                                },
                                {
                                    "text": "dummy_2",
                                    "slot_name": "dummy_slot_name2",
                                    "entity": "dummy_entity_2"
                                },
                                {
                                    "text": " "
                                },
                                {
                                    "text": "at 10p.m.",
                                    "slot_name": "startTime",
                                    "entity": "snips/datetime"
                                },
                                {
                                    "text": " or "
                                },
                                {
                                    "text": "next monday",
                                    "slot_name": "startTime",
                                    "entity": "snips/datetime"
                                }

                            ]
                        }
                    ]
                }
            },
            "language": "en",
            "snips_nlu_version": "0.1.0"
        }

        self.duplicated_utterances_dataset = {
            "entities": {},
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "Hello world"
                                }
                            ]
                        }
                    ]
                },
                "dummy_intent_2": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "Hello world"
                                }
                            ]
                        }
                    ]
                }
            },
            "language": "en",
            "snips_nlu_version": "0.1.0"
        }

        self.slots_dataset = {
            "entities": {
                "dummy_entity_1": {
                    "automatically_extensible": False,
                    "use_synonyms": True,
                    "data": [
                        {
                            "synonyms": [
                                "dummy_a",
                                "dummy 2a",
                                "dummy a",
                                "2 dummy a"
                            ],
                            "value": "dummy_a"
                        },
                        {
                            "synonyms": [
                                "dummy_b",
                                "dummy_bb",
                                "dummy b"
                            ],
                            "value": "dummy_b"
                        },
                        {
                            "synonyms": [
                                "dummy d"
                            ],
                            "value": "dummy d"
                        }
                    ]
                },
                "dummy_entity_2": {
                    "automatically_extensible": False,
                    "use_synonyms": True,
                    "data": [
                        {
                            "synonyms": [
                                "dummy_c",
                                "dummy_cc",
                                "dummy c",
                                "3p.m."
                            ],
                            "value": "dummy_c"
                        }
                    ]
                },
                "snips/datetime": {}
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "This is a "
                                },
                                {
                                    "text": "dummy_1",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                },
                                {
                                    "text": " query with another "
                                },
                                {
                                    "text": "dummy_2",
                                    "slot_name": "dummy_slot_name2",
                                    "entity": "dummy_entity_2"
                                },
                                {
                                    "text": " "
                                },
                                {
                                    "text": "at 10p.m.",
                                    "slot_name": "startTime",
                                    "entity": "snips/datetime"
                                },
                                {
                                    "text": " or "
                                },
                                {
                                    "text": "next monday",
                                    "slot_name": "startTime",
                                    "entity": "snips/datetime"
                                }

                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "This, is, a "
                                },
                                {
                                    "text": "dummy_1",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                }
                            ]
                        }
                    ]
                }
            },
            "language": "en",
            "snips_nlu_version": "1.0.1"
        }

    def test_should_get_intent(self):
        # Given
        dataset = validate_and_format_dataset(self.intent_dataset)

        parser = DeterministicIntentParser().fit(dataset)
        text = "this is a first dummy_a query with a second dummy_c at " \
               "10p.m. or at 12p.m."

        # When
        intent = parser.get_intent(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability)

        self.assertEqual(intent, expected_intent)

    def test_should_get_intent_when_filter(self):
        # Given
        dataset = validate_and_format_dataset(
            self.duplicated_utterances_dataset)

        parser = DeterministicIntentParser().fit(dataset)
        text = "Hello world"
        intent_name_1 = "dummy_intent_1"
        intent_name_2 = "dummy_intent_2"

        # When
        res_1 = parser.get_intent(text, [intent_name_1])
        res_2 = parser.get_intent(text, [intent_name_2])

        # Then
        self.assertEqual(res_1[RES_INTENT_NAME], intent_name_1)
        self.assertEqual(res_2[RES_INTENT_NAME], intent_name_2)

    def test_should_get_intent_after_deserialization(self):
        # Given
        dataset = validate_and_format_dataset(self.intent_dataset)

        parser = DeterministicIntentParser().fit(dataset)
        deserialized_parser = DeterministicIntentParser \
            .from_dict(parser.to_dict())
        text = "this is a first dummy_a query with a second dummy_c at " \
               "10p.m. or at 12p.m."

        # When
        intent = deserialized_parser.get_intent(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability)
        self.assertEqual(intent, expected_intent)

    def test_should_get_slots(self):
        # Given
        dataset = self.slots_dataset
        dataset = validate_and_format_dataset(dataset)

        parser = DeterministicIntentParser().fit(dataset)
        texts = [
            (
                "this is a dummy a query with another dummy_c at 10p.m. or at"
                " 12p.m.",
                [
                    _slot(match_range=(10, 17), value="dummy a",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name"),
                    _slot(match_range=(37, 44), value="dummy_c",
                          entity="dummy_entity_2",
                          slot_name="dummy_slot_name2"),
                    _slot(match_range=(45, 54), value="at 10p.m.",
                          entity="snips/datetime",
                          slot_name="startTime"),
                    _slot(match_range=(58, 67), value="at 12p.m.",
                          entity="snips/datetime",
                          slot_name="startTime")
                ]
            ),
            (
                "this, is,, a, dummy a query with another dummy_c at 10pm or "
                "at 12p.m.",
                [
                    _slot(match_range=(14, 21), value="dummy a",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name"),
                    _slot(match_range=(41, 48), value="dummy_c",
                          entity="dummy_entity_2",
                          slot_name="dummy_slot_name2"),
                    _slot(match_range=(49, 56), value="at 10pm",
                          entity="snips/datetime",
                          slot_name="startTime"),
                    _slot(match_range=(60, 69), value="at 12p.m.",
                          entity="snips/datetime",
                          slot_name="startTime")
                ]
            ),
            (
                "this is a dummy b",
                [
                    _slot(match_range=(10, 17), value="dummy b",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name")
                ]
            ),
            (
                " this is a dummy b ",
                [
                    _slot(match_range=(11, 18), value="dummy b",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name")
                ]
            )
        ]

        for text, expected_slots in texts:
            # When
            slots = parser.get_slots(text, intent="dummy_intent_1")
            # Then
            self.assertListEqual(expected_slots, slots)

    def test_should_get_slots_after_deserialization(self):
        # Given
        dataset = self.slots_dataset
        dataset = validate_and_format_dataset(dataset)

        parser = DeterministicIntentParser().fit(dataset)
        deserialized_parser = DeterministicIntentParser \
            .from_dict(parser.to_dict())
        texts = [
            (
                "this is a dummy a query with another dummy_c at 10p.m. or at"
                " 12p.m.",
                [
                    _slot(match_range=(10, 17), value="dummy a",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name"),
                    _slot(match_range=(37, 44), value="dummy_c",
                          entity="dummy_entity_2",
                          slot_name="dummy_slot_name2"),
                    _slot(match_range=(45, 54), value="at 10p.m.",
                          entity="snips/datetime",
                          slot_name="startTime"),
                    _slot(match_range=(58, 67), value="at 12p.m.",
                          entity="snips/datetime",
                          slot_name="startTime")
                ]
            ),
            (
                "this, is,, a, dummy a query with another dummy_c at 10pm or "
                "at 12p.m.",
                [
                    _slot(match_range=(14, 21), value="dummy a",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name"),
                    _slot(match_range=(41, 48), value="dummy_c",
                          entity="dummy_entity_2",
                          slot_name="dummy_slot_name2"),
                    _slot(match_range=(49, 56), value="at 10pm",
                          entity="snips/datetime",
                          slot_name="startTime"),
                    _slot(match_range=(60, 69), value="at 12p.m.",
                          entity="snips/datetime",
                          slot_name="startTime")
                ]
            ),
            (
                "this is a dummy b",
                [
                    _slot(match_range=(10, 17), value="dummy b",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name")
                ]
            ),
            (
                " this is a dummy b ",
                [
                    _slot(match_range=(11, 18), value="dummy b",
                          entity="dummy_entity_1",
                          slot_name="dummy_slot_name")
                ]
            )
        ]

        for text, expected_slots in texts:
            # When
            slots = deserialized_parser.get_slots(text,
                                                  intent="dummy_intent_1")
            # Then
            self.assertListEqual(expected_slots, slots)

    def test_should_be_serializable_before_fitting(self):
        # Given
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_entities=43)
        parser = DeterministicIntentParser(config=config)

        # When
        actual_dict = parser.to_dict()

        # Then
        expected_dict = {
            "unit_name": "deterministic_intent_parser",
            "config": {
                "unit_name": "deterministic_intent_parser",
                "max_queries": 42,
                "max_entities": 43
            },
            "language_code": None,
            "group_names_to_slot_names": None,
            "patterns": None,
            "slot_names_to_entities": None
        }

        self.assertDictEqual(actual_dict, expected_dict)

    @patch("snips_nlu.intent_parser.deterministic_intent_parser"
           ".generate_regexes")
    def test_should_be_serializable(self, mocked_generate_regexes):
        # Given

        # pylint: disable=unused-argument
        def mock_generate_regexes(utterances, joined_entity_utterances,
                                  group_names_to_slot_names, language):
            regexes = [re.compile(r"mocked_regex_%s" % i)
                       for i in range(len(utterances))]
            group_to_slot = {"group_0": "dummy slot name"}
            return regexes, group_to_slot

        # pylint: enable=unused-argument

        mocked_generate_regexes.side_effect = mock_generate_regexes
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_entities=43)
        parser = DeterministicIntentParser(config=config).fit(dataset)

        # When
        actual_dict = parser.to_dict()

        # Then
        expected_dict = {
            "unit_name": "deterministic_intent_parser",
            "config": {
                "unit_name": "deterministic_intent_parser",
                "max_queries": 42,
                "max_entities": 43
            },
            "language_code": "en",
            "group_names_to_slot_names": {
                "group_0": "dummy slot name"
            },
            "patterns": {
                "dummy_intent_1": [
                    "mocked_regex_0",
                    "mocked_regex_1",
                    "mocked_regex_2",
                    "mocked_regex_3"
                ],
                "dummy_intent_2": [
                    "mocked_regex_0"
                ]
            },
            "slot_names_to_entities": {
                "dummy_slot_name": "dummy_entity_1",
                "dummy slot nàme": "dummy_entity_1",
                "dummy_slot_name3": "dummy_entity_2",
                "dummy_slot_name2": "dummy_entity_2"
            }
        }

        self.assertDictEqual(actual_dict, expected_dict)

    def test_should_be_deserializable(self):
        # Given
        parser_dict = {
            "config": {
                "max_queries": 42,
                "max_entities": 43
            },
            "language_code": "en",
            "group_names_to_slot_names": {
                "hello_group": "hello_slot",
                "world_group": "world_slot"
            },
            "patterns": {
                "intent_name": [
                    "(?P<hello_group>hello?)",
                    "(?P<world_group>world$)"
                ]
            },
            "slot_names_to_entities": {
                "hello_slot": "hello_entity",
                "world_slot": "world_entity"
            }
        }

        # When
        parser = DeterministicIntentParser.from_dict(parser_dict)

        # Then
        patterns = {
            "intent_name": [
                "(?P<hello_group>hello?)",
                "(?P<world_group>world$)"
            ]
        }
        group_names_to_slot_names = {
            "hello_group": "hello_slot",
            "world_group": "world_slot"
        }
        slot_names_to_entities = {
            "hello_slot": "hello_entity",
            "world_slot": "world_entity"
        }
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_entities=43)
        expected_parser = DeterministicIntentParser(config=config)
        expected_parser.language = Language.EN
        expected_parser.group_names_to_slot_names = group_names_to_slot_names
        expected_parser.slot_names_to_entities = slot_names_to_entities
        expected_parser.patterns = patterns

        self.assertEqual(parser, expected_parser)

    def test_should_be_deserializable_before_fitting(self):
        # Given
        parser_dict = {
            "config": {
                "max_queries": 42,
                "max_entities": 43
            },
            "language_code": None,
            "group_names_to_slot_names": None,
            "patterns": None,
            "slot_names_to_entities": None
        }

        # When
        parser = DeterministicIntentParser.from_dict(parser_dict)

        # Then
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_entities=43)
        expected_parser = DeterministicIntentParser(config=config)
        self.assertEqual(parser, expected_parser)

    def test_should_deduplicate_overlapping_slots(self):
        # Given
        language = Language.EN
        slots = [
            _slot(
                [3, 7],
                "non_overlapping1",
                "e",
                "s1"
            ),
            _slot(
                [9, 16],
                "aaaaaaa",
                "e1",
                "s2"
            ),
            _slot(
                [10, 18],
                "bbbbbbbb",
                "e1",
                "s3"
            ),
            _slot(
                [17, 23],
                "b cccc",
                "e1",
                "s4"
            ),
            _slot(
                [50, 60],
                "non_overlapping2",
                "e",
                "s5"
            ),
        ]

        # When
        deduplicated_slots = deduplicate_overlapping_slots(slots, language)

        # Then
        expected_slots = [
            _slot(
                [3, 7],
                "non_overlapping1",
                "e",
                "s1"
            ),
            _slot(
                [17, 23],
                "b cccc",
                "e1",
                "s4"
            ),
            _slot(
                [50, 60],
                "non_overlapping2",
                "e",
                "s5"
            ),
        ]
        self.assertSequenceEqual(deduplicated_slots, expected_slots)

    def test_should_not_train_intents_too_big(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        config = DeterministicIntentParserConfig(max_queries=2,
                                                 max_entities=200)

        # When
        parser = DeterministicIntentParser(config=config).fit(dataset)

        # Then
        not_fitted_intent = "dummy_intent_1"
        fitted_intent = "dummy_intent_2"
        self.assertGreater(len(parser.regexes_per_intent[fitted_intent]), 0)
        self.assertListEqual(parser.regexes_per_intent[not_fitted_intent], [])

    @patch('snips_nlu.intent_parser.deterministic_intent_parser'
           '.get_builtin_entities')
    def test_should_replace_builtin_entities(self, mock_get_builtin_entities):
        # Given
        text = "Be the first to be there at 9pm"
        mock_get_builtin_entities.return_value = [
            {
                RES_MATCH_RANGE: (7, 12),
                VALUE: "first",
                ENTITY: BuiltInEntity.ORDINAL
            },
            {
                RES_MATCH_RANGE: (28, 31),
                VALUE: "9pm",
                ENTITY: BuiltInEntity.DATETIME
            }
        ]

        # When
        range_mapping, processed_text = replace_builtin_entities(
            text=text, language=Language.EN)

        # Then
        expected_mapping = {
            (7, 21): (7, 12),
            (37, 52): (28, 31)
        }
        expected_processed_text = \
            "Be the %SNIPSORDINAL% to be there at %SNIPSDATETIME%"

        self.assertDictEqual(expected_mapping, range_mapping)
        self.assertEqual(expected_processed_text, processed_text)

    def test_should_preprocess_builtin_entities(self):
        # Given
        language = Language.EN
        utterance = {
            DATA: [
                {
                    TEXT: "Raise temperature in room number one to "
                },
                {
                    TEXT: "five degrees",
                    SLOT_NAME: "room_temperature",
                    ENTITY: "temperature"
                },
                {
                    TEXT: " at "
                },
                {
                    TEXT: "9pm",
                    SLOT_NAME: "time",
                    ENTITY: "snips/datetime"
                },
            ]
        }

        # When

        processed_utterance = preprocess_builtin_entities(utterance, language)

        # Then
        expected_utterance = {
            DATA: [
                {
                    TEXT: "Raise temperature in room number %SNIPSNUMBER% to "
                },
                {
                    TEXT: "%SNIPSTEMPERATURE%",
                    SLOT_NAME: "room_temperature",
                    ENTITY: "temperature"
                },
                {
                    TEXT: " at "
                },
                {
                    TEXT: "%SNIPSDATETIME%",
                    SLOT_NAME: "time",
                    ENTITY: "snips/datetime"
                },
            ]
        }

        self.assertDictEqual(expected_utterance, processed_utterance)
