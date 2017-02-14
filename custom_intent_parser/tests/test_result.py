import unittest

from custom_intent_parser.result import parsed_entity, result


class TestResult(unittest.TestCase):
    def test_parsed_entity_can_have_extra_field(self):
        # Given
        rng = (2, 3)
        value = "a"
        entity = "b"
        kwarg = {"c": "c"}
        # When
        parsed_ent = parsed_entity(rng, value, entity, **kwarg)
        # Then
        expected = {
            "range": rng,
            "value": value,
            "entity": entity,
            "c": "c"
        }
        self.assertEqual(parsed_ent, expected)

    def test_entities_should_have_mandatory_fields(self):
        # Given
        rng = (2, 3)
        value = "a"
        parsed_ent = {
            "range": rng,
            "value": value
        }
        text = "a"
        # When/Then
        with self.assertRaises(LookupError) as ctx:
            result(text, entities=[parsed_ent])
        self.assertEqual(ctx.exception.message,
                         "Missing 'entity' key")


if __name__ == '__main__':
    unittest.main()
