import os
import unittest

from snips_nlu.dataset import Dataset
from snips_nlu.entity import Entity
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.tests.utils import TEST_PATH


def get_queries():
    queries = {
        "dummy_intent_1": [
            {
                "data":
                    [
                        {
                            "text": "This is a "
                        },
                        {
                            "text": "dummy_1",
                            "entity": "dummy_entity_1",
                            "slotName": "dummy_slotName"
                        },
                        {
                            "text": " query with another "
                        },
                        {
                            "text": "dummy_2",
                            "entity": "dummy_entity_2",
                            "slotName": "dummy_slotName2"
                        }
                    ]
            },
            {
                "data":
                    [
                        {
                            "text": "This is another "
                        },
                        {
                            "text": "dummy_2_again",
                            "entity": "dummy_entity_2",
                            "slotName": "dummy_slotName2"
                        },
                        {
                            "text": " query."
                        }
                    ]
            },
            {
                "data":
                    [
                        {
                            "text": "This is another "
                        },
                        {
                            "text": "dummy_2_again",
                            "entity": "dummy_entity_2",
                            "slotName": "dummy_slotName3"
                        },
                        {
                            "text": "?"
                        }
                    ]
            },
            {
                "data":
                    [
                        {
                            "text": "dummy_1",
                            "entity": "dummy_entity_1",
                            "slotName": "dummy_slotName"
                        }
                    ]
            }
        ],
        "dummy_intent_2": [
            {
                "data":
                    [
                        {
                            "text": "This is a "
                        },
                        {
                            "text": "dummy_3",
                            "entity": "dummy_entity_1",
                            "slotName": "dummy_slotName"
                        },
                        {
                            "text": " query from another intent"
                        }
                    ]
            }
        ]
    }
    return queries


def get_entities():
    entries_1 = [
        {
            "value": "dummy_a",
            "synonyms": ["dummy_a", "dummy 2a", "dummy a", "2 dummy a"]
        },
        {
            "value": "dummy_b",
            "synonyms": ["dummy_b", "dummy_bb", "dummy b"]
        },
        {
            "value": "dummy\d",
            "synonyms": ["dummy\d"]
        },
    ]
    entity_1 = Entity("dummy_entity_1", entries=entries_1)

    entries_2 = [
        {
            "value": "dummy_c",
            "synonyms": ["dummy_c", "dummy_cc", "dummy c", "3p.m."]
        }
    ]
    entity_2 = Entity("dummy_entity_2", entries=entries_2)
    return {entity_1.name: entity_1, entity_2.name: entity_2}


class TestRegexIntentParser(unittest.TestCase):
    _dataset = None
    _save_path = os.path.join(TEST_PATH, "regex_intent_parser.pkl")

    def setUp(self):
        self._dataset = Dataset(queries=get_queries(),
                                entities=get_entities())

    def tearDown(self):
        if os.path.exists(self._save_path):
            os.remove(self._save_path)

    def test_should_parse(self):
        # Given
        parser = RegexIntentParser("dummy_intent_1").fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        parse = parser.parse(text)

        # Then
        expected_entities = [
            {
                "range": (10, 17),
                "value": "dummy_a",
                "entity": "dummy_entity_1",
                "slotName": "dummy_slotName"
            },
            {
                "range": (37, 44),
                "value": "dummy_c",
                "entity": "dummy_entity_2",
                "slotName": "dummy_slotName2"
            }
        ]
        expected_parse = {
            "text": text,
            "intent": {"name": "dummy_intent_1",
                       "prob": (len("dummy_a") + len("dummy_c")) / float(
                           len(text))},
            "entities": expected_entities
        }
        self.assertEqual(parse["text"], expected_parse["text"])
        self.assertEqual(parse["intent"], expected_parse["intent"])
        self.assertItemsEqual(parse["entities"], expected_parse["entities"])

    def test_should_get_intent(self):
        # Given
        parser = RegexIntentParser("dummy_intent_1").fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        intent = parser.get_intent(text)

        # Then
        expected_intent = {
            "intent": "dummy_intent_1",
            "prob": (len("dummy_a") + len("dummy_c")) / float(len(text))
        }

        self.assertEqual(intent, expected_intent)

    def test_should_get_entities(self):
        # Given
        parser = RegexIntentParser("dummy_intent_1").fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        entities = parser.get_entities(text)

        # Then
        expected_entities = [
            {
                "range": (10, 17),
                "value": "dummy_a",
                "entity": "dummy_entity_1",
                "slotName": "dummy_slotName"
            },
            {
                "range": (37, 44),
                "value": "dummy_c",
                "entity": "dummy_entity_2",
                "slotName": "dummy_slotName2"
            }
        ]
        self.assertItemsEqual(expected_entities, entities)

    def test_save_and_load(self):
        # Given
        parser = RegexIntentParser("dummy_intent_1").fit(self._dataset)

        # When
        parser.save(self._save_path)
        new_parser = RegexIntentParser.load(self._save_path)

        # Then
        self.assertEqual(parser, new_parser)


if __name__ == '__main__':
    unittest.main()
