import io
import os
import shutil
import unittest

from custom_intent_parser.data_helpers import dataset_from_text_file
from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity import Entity
from utils import TEST_PATH


class TestDataHelpers(unittest.TestCase):
    _text_file_path = os.path.join(TEST_PATH, "text_dataset")
    _dataset_path = os.path.join(TEST_PATH, "dataset")

    def setUp(self):
        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

        file_content = u"""
        intent_1 an intent with a @entity_1 and another @entity_2:with_a_role !
        intent_2 another intent with not entity
        """
        with io.open(self._text_file_path, "w", encoding="utf8") as f:
            f.write(file_content)

    def tearDown(self):
        if os.path.exists(self._text_file_path):
            os.remove(self._text_file_path)
        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

    def test_dataset_from_text_file(self):
        # Given
        path = self._text_file_path
        dataset_path = self._dataset_path
        queries = {
            "intent_1": [
                {
                    "data": [
                        {
                            "text": "an intent with a "
                        },
                        {
                            "text": "put_a_entity_1_here",
                            "entity": "entity_1"
                        },
                        {
                            "text": " and another "
                        },
                        {
                            "text": "put_a_entity_2_here",
                            "entity": "entity_2",
                            "role": "with_a_role"
                        },
                        {
                            "text": " !"
                        }
                    ]
                }
            ],
            "intent_2": [
                {
                    "data": [
                        {
                            "text": "another intent with not entity"
                        }
                    ]
                }
            ]
        }
        entities = [
            Entity("entity_1",
                   entries=[{"value": "put_a_entity_1_here",
                             "synonyms": ["put_a_entity_1_here"]}]),
            Entity("entity_2",
                   entries=[{"value": "put_a_entity_2_here",
                             "synonyms": ["put_a_entity_2_here"]}])
        ]
        expected_dataset = Dataset(entities=entities, queries=queries)
        # When
        dataset_from_text_file(path, dataset_path)
        dataset = Dataset.load(dataset_path)
        # Then
        self.assertEqual(dataset, expected_dataset)


if __name__ == '__main__':
    unittest.main()
