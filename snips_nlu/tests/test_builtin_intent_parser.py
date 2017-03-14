import unittest

from ..built_in_intents import BuiltInIntent
from ..intent_parser.builtin_intent_parser import BuiltinIntentParser


class TestBuiltinIntentParser(unittest.TestCase):
    def test_should_parse_intent(self):
        # Given
        candidate_intents = list(BuiltInIntent)
        intent_parser = BuiltinIntentParser(candidate_intents)
        text = "Book me an italian restaurant in NY"

        # When
        intent = intent_parser.get_intent(text)

        # Then
        expected_intent_name = "BookRestaurant"
        self.assertIsNotNone(intent)
        self.assertEqual(expected_intent_name, intent["name"])

    def test_should_parse_entities(self):
        # Given
        candidate_intents = list(BuiltInIntent)
        intent_parser = BuiltinIntentParser(candidate_intents)
        text = "Book me an italian restaurant in NY for 8pm for 2"
        intent_name = "BookRestaurant"

        # When
        entities = intent_parser.get_entities(text, intent=intent_name)

        # Then
        expected_entities = [
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
        self.assertItemsEqual(expected_entities, entities)


if __name__ == '__main__':
    unittest.main()
