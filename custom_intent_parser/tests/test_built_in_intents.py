import unittest

from custom_intent_parser.built_in_intents import (
    get_built_in_intents, get_built_in_intent_entities, BuiltInIntent)


class TestBuiltInIntents(unittest.TestCase):
    def test_should_parse_intents(self):
        # Given
        texts = {
            "Book me an italian restaurant in NY": [
                {"intent": BuiltInIntent.BookRestaurant,
                 "prob": 0.7474335557693172}
            ],
            "yo dude wassup": []
        }
        candidate_intents = list(BuiltInIntent)
        # When
        parsed_intents = dict((t, get_built_in_intents(t, candidate_intents))
                              for t in texts)
        # Then
        for t in texts:
            self.assertEqual(texts[t], parsed_intents[t])

    def test_should_parse_entities(self):
        # Given
        texts = {
            "Book me an italian restaurant in NY for 8pm for 2": {
                "intent": BuiltInIntent.BookRestaurant,
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
        # When
        parsed_intents = dict(
            (t, get_built_in_intent_entities(t, texts[t]["intent"]))
            for t in texts)

        # Then
        for t in texts:
            self.assertItemsEqual(texts[t]["entities"], parsed_intents[t])


if __name__ == '__main__':
    unittest.main()
