import os
import unittest

from ..dataset import validate_dataset
from ..result import result, intent_classification_result, parsed_entity
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.tests.utils import TEST_PATH, SAMPLE_DATASET


class TestRegexIntentParser(unittest.TestCase):
    _dataset = None
    _save_path = os.path.join(TEST_PATH, "regex_intent_parser.pkl")

    def setUp(self):
        self._dataset = SAMPLE_DATASET
        validate_dataset(self._dataset)

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
            parsed_entity(match_range=(10, 17), value="dummy_a",
                          entity="dummy_entity_1", slot_name="dummy_slot_name"),
            parsed_entity(match_range=(37, 44), value="dummy_c",
                          entity="dummy_entity_2", slot_name="dummy_slot_name2")
        ]
        expected_proba = (len("dummy_a") + len("dummy_c")) / float(len(text))
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1",
            probability=expected_proba)
        expected_parse = result(text=text, parsed_intent=expected_intent,
                                parsed_entities=expected_entities)
        self.assertEqual(parse, expected_parse)

    def test_should_get_intent(self):
        # Given
        parser = RegexIntentParser("dummy_intent_1").fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        intent = parser.get_intent(text)

        # Then
        probability = (len("dummy_a") + len("dummy_c")) / float(len(text))
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability)

        self.assertEqual(intent, expected_intent)

    def test_should_get_entities(self):
        # Given
        parser = RegexIntentParser("dummy_intent_1").fit(self._dataset)
        text = "this is a dummy_a query with another dummy_c"

        # When
        entities = parser.get_entities(text)

        # Then
        expected_entities = [
            parsed_entity(match_range=(10, 17), value="dummy_a",
                          entity="dummy_entity_1", slot_name="dummy_slot_name"),
            parsed_entity(match_range=(37, 44), value="dummy_c",
                          entity="dummy_entity_2", slot_name="dummy_slot_name2")
        ]
        self.assertItemsEqual(expected_entities, entities)

        # def test_save_and_load(self):
        #     # Given
        #     parser = RegexIntentParser("dummy_intent_1").fit(self._dataset)
        #
        #     # When
        #     parser.save(self._save_path)
        #     new_parser = RegexIntentParser.load(self._save_path)
        #
        #     # Then
        #     self.assertEqual(parser, new_parser)


if __name__ == '__main__':
    unittest.main()
