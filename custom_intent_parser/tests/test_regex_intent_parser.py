import unittest

from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity_extractor.regex_entity_extractor import \
    RegexEntityExtractor
from custom_intent_parser.intent_parser.regex_intent_parser import \
    RegexIntentParser
from test_regex_entity_extractor import (get_entities, get_queries)


class TestRegexEntityExtractor(unittest.TestCase):
    _dataset = None

    def setUp(self):
        self._dataset = Dataset(queries=get_queries(),
                                entities=get_entities())

    def test_parse_entities(self):
        # Given
        entity_extractor = RegexEntityExtractor().fit(self._dataset)
        parser = RegexIntentParser(entity_extractor).fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        parse = parser.parse(text)

        # Then
        expected_entities = [
            {
                "range": (10, 17),
                "value": "dummy_a",
                "entity": "dummy_entity_1",
                "intent": "dummy_intent_1"
            },
            {
                "range": (37, 44),
                "value": "dummy_c",
                "entity": "dummy_entity_2",
                "intent": "dummy_intent_1"
            }
        ]
        expected_parse = {
            "text": text,
            "intent": {"name": "dummy_intent_1", "prob": 1.0},
            "entities": expected_entities
        }
        self.assertEqual(parse["text"], expected_parse["text"])
        self.assertEqual(parse["intent"], expected_parse["intent"])
        self.assertItemsEqual(parse["entities"], expected_parse["entities"])

    def test_get_entities(self):
        # Given
        entity_extractor = RegexEntityExtractor().fit(self._dataset)
        parser = RegexIntentParser(entity_extractor).fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        entities = parser.get_entities(text)

        # Then
        expected_entities = {
            "entities": [
                {
                    "range": (10, 17),
                    "value": "dummy_a",
                    "entity": "dummy_entity_1",
                    "intent": "dummy_intent_1"
                },
                {
                    "range": (37, 44),
                    "value": "dummy_c",
                    "entity": "dummy_entity_2",
                    "intent": "dummy_intent_1"
                }
            ],
            "text": text
        }
        self.assertEqual(expected_entities["text"], entities["text"])
        self.assertItemsEqual(expected_entities["entities"],
                              entities["entities"])

    def test_get_intent(self):
        # Given
        entity_extractor = RegexEntityExtractor().fit(self._dataset)
        parser = RegexIntentParser(entity_extractor).fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        intent = parser.get_intent(text)

        # Then
        expected_intent = {
            "text": text,
            "intent": {
                "name": "dummy_intent_1",
                "prob": 1.0
            }
        }
        self.assertEqual(intent, expected_intent)


if __name__ == '__main__':
    unittest.main()
