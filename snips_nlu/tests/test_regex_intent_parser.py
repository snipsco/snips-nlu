from __future__ import unicode_literals

import io
import json
import os
import unittest

from snips_nlu.intent_parser.regex_intent_parser import (
    RegexIntentParser, deduplicate_overlapping_slots)
from snips_nlu.languages import Language
from snips_nlu.result import IntentClassificationResult, ParsedSlot
from snips_nlu.tests.utils import TEST_PATH


class TestRegexIntentParser(unittest.TestCase):
    def setUp(self):
        fixtures_directory = os.path.join(TEST_PATH, "fixtures",
                                          "rule_based_parser")
        self.expected_parser_path = os.path.join(fixtures_directory,
                                                 "expected_config.json")
        self.actual_parser_path = os.path.join(fixtures_directory,
                                               "actual_config.json")

    def tearDown(self):
        if os.path.exists(self.actual_parser_path):
            os.remove(self.actual_parser_path)

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

    def test_should_be_saveable(self):
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
        parser.save(self.actual_parser_path)

        # Then
        with io.open(self.expected_parser_path) as f:
            expected_dict = json.load(f)
        with io.open(self.actual_parser_path) as f:
            actual_dict = json.load(f)

        self.assertDictEqual(actual_dict, expected_dict)

    def test_should_be_loadable(self):
        # When
        parser = RegexIntentParser.load(self.expected_parser_path)

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
