import unittest

from snips_nlu.intent_parser.intent_parser import IntentParser

from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
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
        parser = IntentParser.from_dict(parser_dict)
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
        self.assertEqual(parser, expected_parser)

    if __name__ == '__main__':
        unittest.main()
