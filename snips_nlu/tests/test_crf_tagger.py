from __future__ import unicode_literals

import unittest

from mock import patch, MagicMock, Mock

from snips_nlu.config import CRFFeaturesConfig
from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_tagger import CRFTagger, get_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.tokenization import tokenize


class TestCRFTagger(unittest.TestCase):
    @patch('snips_nlu.slot_filler.crf_tagger.serialize_crf_model')
    def test_should_be_serializable(self, mock_serialize_crf_model):
        language = Language.EN
        # Given
        mock_serialize_crf_model.return_value = "mocked_crf_model_data"
        crf_model = get_crf_model()
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
        tagging_scheme = TaggingScheme.BILOU
        data = [
            {
                "tokens": tokenize("I love blue birds", language),
                "tags": ["O", "O", "B-COLOR", "O"]
            },
            {
                "tokens": tokenize("I like red birds", language),
                "tags": ["O", "O", "B-COLOR", "O"]
            }
        ]

        tagger = CRFTagger(crf_model, features_signatures, tagging_scheme,
                           Language.EN)
        tagger.fit(data)

        # When
        actual_tagger_dict = tagger.to_dict()

        # Then
        expected_tagger_dict = {
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
            "tagging_scheme": 2
        }
        self.assertDictEqual(actual_tagger_dict, expected_tagger_dict)

    @patch('snips_nlu.slot_filler.crf_tagger.deserialize_crf_model')
    def test_should_be_deserializable(self, mock_deserialize_crf_model):
        # Given
        language = Language.EN
        mock_deserialize_crf_model.return_value = None
        tagger_dict = {
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
            "tagging_scheme": 2
        }
        # When
        tagger = CRFTagger.from_dict(tagger_dict)

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
        expected_tagging_scheme = TaggingScheme.BILOU
        expected_language = Language.EN

        self.assertListEqual(tagger.features_signatures,
                             expected_features_signatures)
        self.assertEqual(tagger.tagging_scheme, expected_tagging_scheme)
        self.assertEqual(tagger.language, expected_language)

    @patch('snips_nlu.slot_filler.crf_tagger.random')
    def test_should_compute_features(self, mocked_random):
        # Given
        mocked_random.side_effect = [0.9, 0.1, 0.2, 0.4]

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
        crf_features_config = CRFFeaturesConfig(features_drop_out=drop_out)
        tagger = CRFTagger(default_crf_model(), features_signatures,
                           TaggingScheme.BIO, Language.EN, crf_features_config)

        tokens = tokenize("foo hello world bar", Language.EN)

        # When
        features_with_drop_out = tagger.compute_features(tokens, True)

        # Then
        expected_features = [
            {"ngram_1": "foo"},
            {},
            {},
            {"ngram_1": "bar"}
        ]
        self.assertListEqual(expected_features, features_with_drop_out)
