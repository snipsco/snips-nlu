import os
import unittest
from shutil import rmtree

from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity import Entity
from utils import TEST_PATH


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
                            "role": "dummy_role"
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
                            "text": "dummy_2",
                            "entity": "dummy_entity_2"
                        },
                        {
                            "text": " query."
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
                            "entity": "dummy_entity_1"
                        },
                        {
                            "text": " query from another intent"
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
                            "text": "dummy_4",
                            "entity": "dummy_entity_2",
                            "role": "dummy_role"
                        },
                        {
                            "text": " query from another intent"
                        }
                    ]
            }
        ]
    }
    return queries


class TestDataset(unittest.TestCase):
    dataset_dir = os.path.join(TEST_PATH, "dataset")

    def tearDown(self):
        if os.path.exists(self.dataset_dir):
            rmtree(self.dataset_dir)

    def test_dataset_should_save_and_reload(self):
        # Given
        entities = {
            "dummy_entity_1": Entity("dummy_entity_1"),
            "dummy_entity_2": Entity("dummy_entity_2")
        }
        queries = get_queries()

        # When
        dataset = Dataset(entities=entities, queries=queries)
        dataset.save(self.dataset_dir)
        new_dataset = Dataset.load(self.dataset_dir)

        # Then
        self.assertEqual(dataset, new_dataset)

    def test_invalid_intent_name_should_raise_exception(self):
        # Given
        entities = {
            "dummy_entity_1": Entity("dummy_entity_1"),
            "dummy_entity_2": Entity("dummy_entity_2")
        }
        queries = get_queries()
        queries["invalid/intent_name"] = []

        # When/Then
        with self.assertRaises(ValueError) as ctx:
            Dataset(entities=entities, queries=queries)
        self.assertEqual(ctx.exception.message,
                         "invalid/intent_name is an invalid intent name. "
                         "Intent names must be a valid file name, use only: "
                         "[a-zA-Z0-9_- ]")

    def test_unknown_entity_should_raise_exception(self):
        # Given
        entities = {
            "dummy_entity_1": Entity("dummy_entity_1")
        }
        queries = get_queries()

        # When/Then
        with self.assertRaises(ValueError) as ctx:
            Dataset(entities=entities, queries=queries)
        self.assertEqual(ctx.exception.message,
                         "Unknown entity 'dummy_entity_2'. Entities must"
                         " belong to ['dummy_entity_1']")


if __name__ == '__main__':
    unittest.main()
