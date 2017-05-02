import json
import unittest

from snips_nlu.intent_parser.regex_intent_parser import (
    RegexIntentParser, deduplicate_overlapping_slots)
from snips_nlu.result import IntentClassificationResult, ParsedSlot
from snips_nlu.tests.utils import SAMPLE_DATASET


class TestRegexIntentParser(unittest.TestCase):
    def test_should_get_intent(self):
        # Given
        dataset = SAMPLE_DATASET
        parser = RegexIntentParser().fit(dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        intent = parser.get_intent(text)

        # Then
        probability = 1.0
        expected_intent = IntentClassificationResult(
            intent_name="dummy_intent_1", probability=probability)

        self.assertEqual(intent, expected_intent)

    def test_should_get_slots(self):
        # Given
        dataset = SAMPLE_DATASET
        parser = RegexIntentParser().fit(dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        slots = parser.get_slots(text, intent="dummy_intent_1")

        # Then
        expected_slots = [
            ParsedSlot(match_range=(10, 17), value="dummy_a",
                       entity="dummy_entity_1", slot_name="dummy_slot_name"),
            ParsedSlot(match_range=(37, 44), value="dummy_c",
                       entity="dummy_entity_2", slot_name="dummy_slot_name2")
        ]
        self.assertItemsEqual(expected_slots, slots)

    def test_should_be_serializable(self):
        # Given
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
            patterns=patterns,
            group_names_to_slot_names=group_names_to_slot_names,
            slot_names_to_entities=slot_names_to_entities
        )

        # When
        parser_dict = parser.to_dict()

        # Then
        # noinspection PyBroadException
        try:
            json.dumps(parser_dict).decode("utf-8")
        except:
            self.fail("Parser dict should be json serializable in utf-8")

        expected_dict = {
            "@class_name": "RegexIntentParser",
            "@module_name": "snips_nlu.intent_parser.regex_intent_parser",
            'group_names_to_slot_names': {
                'hello_group': 'hello_slot',
                'world_group': 'world_slot'
            },
            'patterns': {
                'intent_name': [
                    '(?P<hello_group>hello?)',
                    '(?P<world_group>world$)'
                ]
            },
            'slot_names_to_entities': {
                'hello_slot': 'hello_entity',
                'world_slot': 'world_entity'
            }
        }
        self.assertDictEqual(parser_dict, expected_dict)

    def test_should_be_deserializable(self):
        # Given
        parser_dict = {
            "@class_name": "RegexIntentParser",
            "@module_name": "snips_nlu.intent_parser.regex_intent_parser",
            'group_names_to_slot_names': {
                'hello_group': 'hello_slot',
                'world_group': 'world_slot'
            },
            'patterns': {
                'intent_name': [
                    '(?P<hello_group>hello?)',
                    '(?P<world_group>world$)'
                ]
            },
            'slot_names_to_entities': {
                'hello_slot': 'hello_entity',
                'world_slot': 'world_entity'
            }
        }

        # When
        parser = RegexIntentParser.from_dict(parser_dict)
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
            patterns=patterns,
            group_names_to_slot_names=group_names_to_slot_names,
            slot_names_to_entities=slot_names_to_entities
        )

        # noinspection PyBroadException
        try:
            parser_json = json.dumps(parser_dict).decode("utf-8")
            _ = RegexIntentParser.from_dict(json.loads(parser_json))
        except:
            self.fail("RegexIntentParser should be deserializable from dict "
                      "with unicode values")

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
