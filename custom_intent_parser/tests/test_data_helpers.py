import io
import json
import os
import shutil
import unittest

from custom_intent_parser.data_helpers import dataset_from_asset_directory
from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity import Entity
from utils import TEST_PATH


class TestDataHelpers(unittest.TestCase):
    _nlu_assets_path = os.path.join(TEST_PATH, "nlu_assets")
    _sample_utterance_path = os.path.join(_nlu_assets_path,
                                          "SamplesUtterances.txt")
    _ontology_path = os.path.join(_nlu_assets_path, "DummyIntent.json")
    _entity_1_path = os.path.join(_nlu_assets_path, "entity_1.txt")
    _entity_2_path = os.path.join(_nlu_assets_path, "entity_2.txt")
    _dataset_path = os.path.join(TEST_PATH, "dataset")

    def setUp(self):
        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

        if os.path.exists(self._nlu_assets_path):
            shutil.rmtree(self._nlu_assets_path)

        os.mkdir(self._nlu_assets_path)

        utterances = u"""
        intent_1 an intent with a {entity_1_name} and another {entity_2_name} !
        intent_1 another intent with {entity_2_other_name} !
        intent_2 another intent with not entity
        """
        with io.open(self._sample_utterance_path, "w", encoding="utf8") as f:
            f.write(utterances)

        intent_content = u"""
        {
            "intents": [
                {
                    "intent": "intent_1",
                    "slots": [
                        {"name": "entity_1_name", "entityName": "entity_1"},
                        {"name": "entity_2_name", "entityName": "entity_2"},
                        {"name": "entity_2_other_name",
                            "entityName": "entity_2"}
                        ]
                },
                {
                    "intent": "intent_2",
                    "slots": []
                }
            ],
            "entities": [
                {
                    "entityName": "entity_1",
                    "automaticallyExtensible": false,
                    "useSynonyms": false
                },
                {
                    "entityName": "entity_2",
                    "automaticallyExtensible": true,
                    "useSynonyms": true
                }
            ]
        }
        """
        with io.open(self._ontology_path, "w", encoding="utf8") as f:
            json_data = json.loads(intent_content)
            data = json.dumps(json_data)
            f.write(unicode(data))

        entity_1 = u"""my_entity_1
        my_other_entity_1
        """
        with io.open(self._entity_1_path, "w", encoding="utf8") as f:
            f.write(entity_1)

        entity_2 = u"""my_entity_2;my_entity_2_synonym
        my_other_entity_2
        """
        with io.open(self._entity_2_path, "w", encoding="utf8") as f:
            f.write(entity_2)

    def tearDown(self):
        if os.path.exists(self._nlu_assets_path):
            shutil.rmtree(self._nlu_assets_path)
        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

    def test_dataset_from_asset_directory(self):
        # Given
        path = self._nlu_assets_path
        dataset_path = self._dataset_path
        queries = {
            "intent_1": [
                {
                    "data": [
                        {
                            "text": "an intent with a "
                        },
                        {
                            "text": "dummy_entity_1",
                            "entity": "entity_1",
                            "role": "entity_1_name"
                        },
                        {
                            "text": " and another "
                        },
                        {
                            "text": "dummy_entity_2",
                            "entity": "entity_2",
                            "role": "entity_2_name"
                        },
                        {
                            "text": " !"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "another intent with "
                        },
                        {
                            "text": "dummy_entity_2",
                            "entity": "entity_2",
                            "role": "entity_2_other_name"
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
                   entries=[
                       {"value": "my_entity_1",
                        "synonyms": ["my_entity_1"]},
                       {"value": "my_other_entity_1",
                        "synonyms": ["my_other_entity_1"]}
                   ]),
            Entity("entity_2", use_learning=True, use_synonyms=True,
                   entries=[
                       {"value": "my_entity_2",
                        "synonyms": ["my_entity_2", "my_entity_2_synonym"]},
                       {"value": "my_other_entity_2",
                        "synonyms": ["my_other_entity_2"]}
                   ])
        ]
        expected_dataset = Dataset(entities=entities, queries=queries)
        # When
        dataset_from_asset_directory(path, dataset_path)
        dataset = Dataset.load(dataset_path)
        # Then
        self.assertEqual(dataset, expected_dataset)


if __name__ == '__main__':
    unittest.main()
