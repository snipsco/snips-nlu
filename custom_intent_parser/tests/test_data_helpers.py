import io
import json
import os
import shutil
import unittest

from custom_intent_parser.data_helpers import dataset_from_asset_directories
from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity import Entity
from utils import TEST_PATH


class TestDataHelpers(unittest.TestCase):
    _nlu_assets_path_1 = os.path.join(TEST_PATH, "nlu_assets_1")
    _sample_utterance_path_1 = os.path.join(_nlu_assets_path_1,
                                            "SamplesUtterances.txt")
    _ontology_path_1 = os.path.join(_nlu_assets_path_1, "DummyIntent.json")
    _entity_1_path = os.path.join(_nlu_assets_path_1, "entity_1.txt")
    _entity_2_path = os.path.join(_nlu_assets_path_1, "entity_2.txt")

    _nlu_assets_path_2 = os.path.join(TEST_PATH, "nlu_assets_2")
    _sample_utterance_path_2 = os.path.join(_nlu_assets_path_2,
                                            "SamplesUtterances.txt")
    _ontology_path_2 = os.path.join(_nlu_assets_path_2,
                                    "OtherDummyIntent.json")
    _entity_3_path = os.path.join(_nlu_assets_path_2, "entity_3.txt")
    _entity_4_path = os.path.join(_nlu_assets_path_2, "entity_4.txt")

    _dataset_path = os.path.join(TEST_PATH, "dataset")

    def setUp(self):
        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

        if os.path.exists(self._nlu_assets_path_1):
            shutil.rmtree(self._nlu_assets_path_1)
        os.mkdir(self._nlu_assets_path_1)

        if os.path.exists(self._nlu_assets_path_2):
            shutil.rmtree(self._nlu_assets_path_2)
        os.mkdir(self._nlu_assets_path_2)

        utterances_1 = u"""
        intent_1 a query with a {entity_1_name} and another {entity_2_name} !
        intent_1 another query with {entity_2_other_name} !
        intent_2 another query with no entity
        """
        with io.open(self._sample_utterance_path_1, "w", encoding="utf8") as f:
            f.write(utterances_1)

        intent_content_1 = u"""
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
        with io.open(self._ontology_path_1, "w", encoding="utf8") as f:
            json_data = json.loads(intent_content_1)
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

        utterances_2 = u"""
        intent_3 a query with a {entity_3_name} and another {entity_4_name} !
        intent_3 another query with {entity_4_other_name} !
        intent_4 another query with no entity
        """
        with io.open(self._sample_utterance_path_2, "w", encoding="utf8") as f:
            f.write(utterances_2)

        intent_content_2 = u"""
        {
            "intents": [
                {
                    "intent": "intent_3",
                    "slots": [
                        {"name": "entity_3_name", "entityName": "entity_3"},
                        {"name": "entity_4_name", "entityName": "entity_4"},
                        {"name": "entity_4_other_name",
                            "entityName": "entity_4"}
                        ]
                },
                {
                    "intent": "intent_4",
                    "slots": []
                }
            ],
            "entities": [
                {
                    "entityName": "entity_3",
                    "automaticallyExtensible": false,
                    "useSynonyms": false
                },
                {
                    "entityName": "entity_4",
                    "automaticallyExtensible": true,
                    "useSynonyms": true
                }
            ]
        }
        """
        with io.open(self._ontology_path_2, "w", encoding="utf8") as f:
            json_data = json.loads(intent_content_2)
            data = json.dumps(json_data)
            f.write(unicode(data))

        entity_3 = u"""my_entity_3
                my_other_entity_3
                """
        with io.open(self._entity_3_path, "w", encoding="utf8") as f:
            f.write(entity_3)

        entity_4 = u"""my_entity_4;my_entity_4_synonym
                my_other_entity_4
                """
        with io.open(self._entity_4_path, "w", encoding="utf8") as f:
            f.write(entity_4)

    def tearDown(self):
        if os.path.exists(self._nlu_assets_path_1):
            shutil.rmtree(self._nlu_assets_path_1)
        if os.path.exists(self._nlu_assets_path_2):
            shutil.rmtree(self._nlu_assets_path_2)
        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

    def test_dataset_from_asset_directories(self):
        # Given
        paths = [self._nlu_assets_path_1, self._nlu_assets_path_2]
        dataset_path = self._dataset_path
        queries = {
            "intent_1": [
                {
                    "data": [
                        {
                            "text": "a query with a "
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
                            "text": "another query with "
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
                            "text": "another query with no entity"
                        }
                    ]
                }
            ],
            "intent_3": [
                {
                    "data": [
                        {
                            "text": "a query with a "
                        },
                        {
                            "text": "dummy_entity_3",
                            "entity": "entity_3",
                            "role": "entity_3_name"
                        },
                        {
                            "text": " and another "
                        },
                        {
                            "text": "dummy_entity_4",
                            "entity": "entity_4",
                            "role": "entity_4_name"
                        },
                        {
                            "text": " !"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "another query with "
                        },
                        {
                            "text": "dummy_entity_4",
                            "entity": "entity_4",
                            "role": "entity_4_other_name"
                        },
                        {
                            "text": " !"
                        }
                    ]
                }
            ],
            "intent_4": [
                {
                    "data": [
                        {
                            "text": "another query with no entity"
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
                   ]),
            Entity("entity_3",
                   entries=[
                       {"value": "my_entity_3",
                        "synonyms": ["my_entity_3"]},
                       {"value": "my_other_entity_3",
                        "synonyms": ["my_other_entity_3"]}
                   ]),
            Entity("entity_4", use_learning=True, use_synonyms=True,
                   entries=[
                       {"value": "my_entity_4",
                        "synonyms": ["my_entity_4", "my_entity_4_synonym"]},
                       {"value": "my_other_entity_4",
                        "synonyms": ["my_other_entity_4"]}
                   ])
        ]
        expected_dataset = Dataset(entities=entities, queries=queries)
        # When
        dataset_from_asset_directories(paths, dataset_path)
        dataset = Dataset.load(dataset_path)
        # Then
        self.assertEqual(dataset, expected_dataset)


if __name__ == '__main__':
    unittest.main()
