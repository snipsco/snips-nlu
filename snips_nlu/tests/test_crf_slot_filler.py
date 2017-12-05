# coding=utf-8
from __future__ import unicode_literals

import unittest

from mock import patch

from snips_nlu.config import CRFSlotFillerConfig
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_slot_filler import CRFSlotFiller
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.tests.utils import SAMPLE_DATASET
from snips_nlu.tokenization import tokenize


class TestCRFSlotFiller(unittest.TestCase):
    @patch('snips_nlu.slot_filler.crf_slot_filler.serialize_crf_model')
    def test_should_be_serializable(self, mock_serialize_crf_model):
        language = Language.EN
        # Given
        mock_serialize_crf_model.return_value = "mocked_crf_model_data"
        features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2, "language_code": language.iso_code},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(tagging_scheme=TaggingScheme.BILOU)
        dataset = validate_and_format_dataset(SAMPLE_DATASET)

        slot_filler = CRFSlotFiller(features_signatures, config)
        intent = "dummy_intent_1"
        slot_filler.fit(dataset, intent=intent)

        # When
        actual_slot_filler_dict = slot_filler.to_dict()

        # Then
        expected_slot_filler_dict = {
            "crf_model_data": "mocked_crf_model_data",
            "features_signatures": [
                {
                    "args": {
                        "n": 1,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language_code": "en",
            "config": config.to_dict(),
            "intent": intent,
            "slot_name_mapping": {
                "dummy_intent_1": {
                    "dummy_slot_name": "dummy_entity_1",
                    "dummy_slot_name2": "dummy_entity_2",
                    "dummy_slot_name3": "dummy_entity_2",
                },
                "dummy_intent_2": {
                    "dummy slot nàme": "dummy_entity_1"
                }
            },
        }
        self.assertDictEqual(actual_slot_filler_dict,
                             expected_slot_filler_dict)

    @patch('snips_nlu.slot_filler.crf_slot_filler.deserialize_crf_model')
    def test_should_be_deserializable(self, mock_deserialize_crf_model):
        # Given
        language = Language.EN
        mock_deserialize_crf_model.return_value = None
        slot_filler_dict = {
            "crf_model_data": "mocked_crf_model_data",
            "features_signatures": [
                {
                    "args": {
                        "n": 1,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language_code": "en",
            "intent": "dummy_intent_1",
            "slot_name_mapping": {
                "dummy_intent_1": {
                    "dummy_slot_name": "dummy_entity_1",
                }
            },
            "config": CRFSlotFillerConfig().to_dict()
        }
        # When
        slot_filler = CRFSlotFiller.from_dict(slot_filler_dict)

        # Then
        mock_deserialize_crf_model.assert_called_once_with(
            "mocked_crf_model_data")
        expected_features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2, "language_code": language.iso_code},
                "offsets": [-1, 0]
            }
        ]
        expected_language = Language.EN
        expected_config = CRFSlotFillerConfig()
        expected_intent = "dummy_intent_1"
        expected_slot_name_mapping = {
            "dummy_intent_1": {
                "dummy_slot_name": "dummy_entity_1",
            }
        }

        self.assertListEqual(slot_filler.features_signatures,
                             expected_features_signatures)
        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping,
                         expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(),
                             slot_filler.config.to_dict())

    def test_should_compute_features(self):
        # Given
        features_signatures = [
            {
                "factory_name": "get_ngram_fn",
                "args": {
                    "n": 1,
                    "use_stemming": False,
                    "language_code": "en",
                    "common_words_gazetteer_name": None
                },
                "offsets": [0]
            },
        ]
        drop_out = {
            "ngram_1": 0.3
        }
        slot_filler_config = CRFSlotFillerConfig(features_drop_out=drop_out,
                                                 random_seed=40)
        slot_filler = CRFSlotFiller(features_signatures, slot_filler_config)

        tokens = tokenize("foo hello world bar", Language.EN)
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        slot_filler.fit(dataset, intent="dummy_intent_1")

        # When
        features_with_drop_out = slot_filler._compute_features(tokens, True)

        # Then
        expected_features = [
            {"ngram_1": "foo"},
            {},
            {"ngram_1": "world"},
            {},
        ]
        self.assertListEqual(expected_features, features_with_drop_out)
