import os
import unittest

from custom_intent_parser.entity import Entity
from utils import TEST_PATH


class TestEntity(unittest.TestCase):
    save_path = os.path.join(TEST_PATH, "saved_entity.json")

    def tearDown(self):
        if os.path.isfile(self.save_path):
            os.remove(self.save_path)

    def test_entity_should_save_and_load(self):
        # Given
        entries = [
            {
                "value": "dummy1",
                "synonyms": ["dummy1"]
            },
            {
                "value": "dummy2",
                "synonyms": ["dummy2"]
            }
        ]
        name = "dummy"
        use_learning = True
        use_synonyms = True

        entity = Entity(name, entries=entries, use_learning=use_learning,
                        use_synonyms=use_synonyms)
        # When
        entity.to_json(self.save_path)
        loaded_entity = Entity.from_json(self.save_path)

        # Then
        self.assertEqual(entity.name, loaded_entity.name)
        self.assertSequenceEqual(entity.entries, loaded_entity.entries)
        self.assertEqual(entity.use_learning, loaded_entity.use_learning)
        self.assertEqual(entity.use_synonyms, loaded_entity.use_synonyms)

    def test_invalid_name_should_raise_exception(self):
        # Given
        name = "dummy:dummier"
        # When / Then
        with self.assertRaises(ValueError) as ctx:
            Entity(name)
        self.assertEqual("Entity name must be alpha numeric,"
                         " found 'dummy:dummier'", ctx.exception.message)

    def test_invalid_synonyms_should_raise_exception(self):
        # Given
        name = "dummy"
        entries = [
            {
                "value": "dummy1",
                "synonyms": []
            }
        ]
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            Entity(name, entries=entries)
        self.assertEqual("There must be at least one synonym equal to "
                         "'value' field", ctx.exception.message)

        # Given
        entries = [
            {
                "value": "dummy1",
                "synonyms": ["dummy2"]
            }
        ]
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            Entity(name, entries=entries)
        self.assertEqual("If there only one synonym it must be equal "
                         "to the 'value' field", ctx.exception.message)

    def test_missing_or_extra_field_should_raise_exception(self):
        # Given
        name = "dummy"
        entries = [
            {
                "value": "dummy1",
            }
        ]
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            Entity(name, entries=entries)
        self.assertEqual("Missing entry keys: set(['synonyms'])",
                         ctx.exception.message)

        # Given
        entries = [
            {
                "synonyms": [],
            }
        ]
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            Entity(name, entries=entries)
        self.assertEqual("Missing entry keys: set(['value'])",
                         ctx.exception.message)

        # Given
        entries = [
            {
                "dummy_key": "dummy_value",
                "synonyms": ["dummy"],
                "value": ["dummy"],
            }
        ]
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            Entity(name, entries=entries)
        self.assertEqual("Unexpected entry keys: set(['dummy_key'])",
                         ctx.exception.message)


if __name__ == '__main__':
    unittest.main()
