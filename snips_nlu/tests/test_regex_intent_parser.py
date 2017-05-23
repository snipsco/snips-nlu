from __future__ import unicode_literals

import unittest

from snips_nlu.constants import INTENTS
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.regex_intent_parser import (
    RegexIntentParser, deduplicate_overlapping_slots)
from snips_nlu.languages import Language
from snips_nlu.result import IntentClassificationResult, ParsedSlot
from snips_nlu.tests.utils import SAMPLE_DATASET


class TestRegexIntentParser(unittest.TestCase):
    def test_should_get_intent(self):
        # Given
        language = Language.EN
        dataset = {
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
                    "engineType": "regex",
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
                        }
                    ]
                }
            },
            "language": language.iso_code
        }

        parser = RegexIntentParser(language).fit(dataset)
        text = "this is a dummy_a query with another dummy_c at 10p.m. or " \
               "at 12p.m."

        # When
        intent = parser.get_intent(text)

        # Then
        probability = 1.0
        expected_intent = IntentClassificationResult(
            intent_name="dummy_intent_1", probability=probability)

        self.assertEqual(intent, expected_intent)

    def test_should_get_slots(self):
        # Given
        language = Language.EN
        dataset = {
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
                    "engineType": "regex",
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
                        }
                    ]
                }
            },
            "language": language.iso_code
        }
        parser = RegexIntentParser(language).fit(dataset)
        text = "this is a dummy_a query with another dummy_c at 10p.m. or " \
               "at 12p.m."

        # When
        slots = parser.get_slots(text, intent="dummy_intent_1")

        # Then
        expected_slots = [
            ParsedSlot(match_range=(10, 17), value="dummy_a",
                       entity="dummy_entity_1", slot_name="dummy_slot_name"),
            ParsedSlot(match_range=(37, 44), value="dummy_c",
                       entity="dummy_entity_2", slot_name="dummy_slot_name2"),
            ParsedSlot(match_range=(45, 54), value="at 10p.m.",
                       entity="snips/datetime", slot_name="startTime"),
            ParsedSlot(match_range=(58, 67), value="at 12p.m.",
                       entity="snips/datetime", slot_name="startTime")
        ]
        self.assertItemsEqual(expected_slots, slots)

    def test_should_be_serializable(self):
        # Given
        language = Language.EN
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
        parser = RegexIntentParser(
            language=language,
            patterns=patterns,
            group_names_to_slot_names=group_names_to_slot_names,
            slot_names_to_entities=slot_names_to_entities
        )

        # When
        actual_dict = parser.to_dict()

        # Then
        expected_dict = {
            "language": "en",
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

        self.assertDictEqual(actual_dict, expected_dict)

    def test_should_be_deserializable(self):
        # Given
        parser_dict = {
            "language": "en",
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
        parser = RegexIntentParser.from_dict(parser_dict)

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
        expected_parser = RegexIntentParser(
            language=Language.EN,
            patterns=patterns,
            group_names_to_slot_names=group_names_to_slot_names,
            slot_names_to_entities=slot_names_to_entities
        )

        self.assertEqual(parser, expected_parser)

    def test_should_deduplicate_overlapping_slots(self):
        # Given
        slots = [
            ParsedSlot(
                [3, 7],
                "non_overlapping1",
                "e",
                "s1"
            ),
            ParsedSlot(
                [9, 16],
                "aaaaaaa",
                "e1",
                "s2"
            ),
            ParsedSlot(
                [10, 18],
                "bbbbbbbb",
                "e1",
                "s3"
            ),
            ParsedSlot(
                [17, 23],
                "b cccc",
                "e1",
                "s4"
            ),
            ParsedSlot(
                [50, 60],
                "non_overlapping2",
                "e",
                "s5"
            ),
        ]

        # When
        deduplicated_slots = deduplicate_overlapping_slots(slots)

        # Then
        expected_slots = [
            ParsedSlot(
                [3, 7],
                "non_overlapping1",
                "e",
                "s1"
            ),
            ParsedSlot(
                [17, 23],
                "b cccc",
                "e1",
                "s4"
            ),
            ParsedSlot(
                [50, 60],
                "non_overlapping2",
                "e",
                "s5"
            ),
        ]
        self.assertSequenceEqual(deduplicated_slots, expected_slots)

    def test_should_train_only_specified_intents(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        language = Language.EN
        intents = ["dummy_intent_1"]

        # When
        parser = RegexIntentParser(language).fit(dataset, intents=intents)

        # Then
        self.assertGreater(len(parser.regexes_per_intent[intents[0]]), 0)
        for intent_name in dataset[INTENTS]:
            if intent_name not in intents:
                self.assertEqual(len(parser.regexes_per_intent[intent_name]),
                                 0)
