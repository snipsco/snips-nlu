from __future__ import unicode_literals

import unittest

from mock import patch

from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.tokenization import tokenize


class TestCRFTagger(unittest.TestCase):
    @patch('snips_nlu.slot_filler.crf_tagger.safe_pickle_dumps')
    def test_should_be_serializable(self, mock_pkl_dumps):
        # Given
        mock_pkl_dumps.return_value = "mocked_crf_pkl"
        crf_model = default_crf_model()
        features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2},
                "offsets": [-1, 0]
            }
        ]
        tagging_scheme = TaggingScheme.BILOU
        data = [
            {
                "tokens": tokenize("I love blue birds"),
                "tags": ["O", "O", "B-COLOR", "O"]
            },
            {
                "tokens": tokenize("I like red birds"),
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
            "crf_model_pkl": "mocked_crf_pkl",
            "features_signatures": [
                {
                    "args": {
                        "n": 1
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language": "en",
            "tagging_scheme": 2
        }
        self.assertDictEqual(actual_tagger_dict, expected_tagger_dict)

    @patch('snips_nlu.slot_filler.crf_tagger.safe_pickle_loads')
    def test_should_be_deserializable(self, mock_pkl_loads):
        # Given
        mock_pkl_loads.return_value = None
        tagger_dict = {
            "crf_model_pkl": "mocked_crf_pkl",
            "features_signatures": [
                {
                    "args": {
                        "n": 1
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language": "en",
            "tagging_scheme": 2
        }
        # When
        tagger = CRFTagger.from_dict(tagger_dict)

        # Then
        mock_pkl_loads.assert_called_once_with("mocked_crf_pkl")
        expected_features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2},
                "offsets": [-1, 0]
            }
        ]
        expected_tagging_scheme = TaggingScheme.BILOU
        expected_language = Language.EN

        self.assertListEqual(tagger.features_signatures,
                             expected_features_signatures)
        self.assertEqual(tagger.tagging_scheme, expected_tagging_scheme)
        self.assertEqual(tagger.language, expected_language)
