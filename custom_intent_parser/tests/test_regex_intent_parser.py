import os
import shutil
import unittest

from custom_intent_parser.built_in_intents import BuiltInIntent
from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity_extractor.regex_entity_extractor import \
    RegexEntityExtractor
from custom_intent_parser.intent_parser.regex_intent_parser import \
    RegexIntentParser
from custom_intent_parser.tests.utils import TEST_PATH
from test_regex_entity_extractor import (get_entities, get_queries)


class TestRegexEntityExtractor(unittest.TestCase):
    _dataset = None
    _save_path = os.path.join(TEST_PATH, "regex_intent_parser")

    def setUp(self):
        self._dataset = Dataset(queries=get_queries(),
                                entities=get_entities())

    def tearDown(self):
        if os.path.exists(self._save_path):
            shutil.rmtree(self._save_path)

    def test_parse(self):
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
                "slotName": "dummy_slotName"
            },
            {
                "range": (37, 44),
                "value": "dummy_c",
                "entity": "dummy_entity_2"
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
                "entity": "dummy_entity_2"
            }
        ]
        self.assertItemsEqual(expected_entities, entities)

    def test_get_intent(self):
        # Given
        entity_extractor = RegexEntityExtractor().fit(self._dataset)
        parser = RegexIntentParser(entity_extractor).fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        intent = parser.get_intent(text)

        # Then
        expected_intent = {
            "intent": "dummy_intent_1",
            "prob": 1.0
        }
        self.assertEqual(intent, expected_intent)

        if __name__ == '__main__':
            unittest.main()

    def test_should_get_built_in(self):
        # Given
        entity_extractor = RegexEntityExtractor().fit(self._dataset)
        built_in_intents = [BuiltInIntent.BookRestaurant]
        parser = RegexIntentParser(entity_extractor,
                                   built_in_intents=built_in_intents)
        parser = parser.fit(self._dataset)
        texts = {
            "Book me an italian restaurant in NY for 8pm for 2": {
                "text": "Book me an italian restaurant in NY for 8pm for 2",
                "intent": {
                    "name": BuiltInIntent.BookRestaurant.value["name"],
                    "prob": 0.9794680411508389
                },
                "entities": [
                    {
                        u"value": u"an italian restaurant in NY",
                        u"range": (8, 35),
                        u"entity": u"restaurant"
                    },
                    {
                        u"value": u"2",
                        u"range": (48, 49),
                        u"entity": u"partySize"
                    },
                    {
                        u"value": u"for 8pm",
                        u"range": (36, 43),
                        u"entity": u"reservationDatetime"
                    }

                ]
            }
        }

        # When / Then
        for text, expected_parse in texts.iteritems():
            parse = parser.parse(text)
            self.assertEqual(parse["text"], expected_parse["text"])
            self.assertEqual(parse["intent"], expected_parse["intent"])
            self.assertItemsEqual(parse["entities"],
                                  expected_parse["entities"])

    def test_save_and_load(self):
        # Given
        entity_extractor = RegexEntityExtractor().fit(self._dataset)
        parser = RegexIntentParser(entity_extractor).fit(self._dataset)

        # When
        parser.save(self._save_path)
        new_parser = RegexIntentParser.load(self._save_path)

        self.assertEqual(parser.entity_extractor.group_names_to_labels,
                         new_parser.entity_extractor.group_names_to_labels)
        for intent_name, patterns \
                in parser.entity_extractor.regexes.iteritems():
            self.assertEqual(
                [r.pattern for r in patterns],
                [r.pattern for r in
                 new_parser.entity_extractor.regexes[intent_name]])
            self.assertEqual(
                [r.flags for r in patterns],
                [r.flags for r in
                 new_parser.entity_extractor.regexes[intent_name]])
        self.assertEqual(parser._cache, new_parser._cache)


if __name__ == '__main__':
    unittest.main()
