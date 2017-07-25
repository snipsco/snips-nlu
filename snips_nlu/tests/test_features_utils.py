import unittest

from snips_nlu.constants import NGRAM, TOKEN_INDEXES
from snips_nlu.slot_filler.features_utils import get_all_ngrams


class TestFeaturesUtils(unittest.TestCase):
    def test_get_all_ngrams(self):
        # Given
        tokens = ["this", "is", "a", "simple", "sentence"]

        # When
        ngrams = get_all_ngrams(tokens)

        # Then
        expected_ngrams = [
            {NGRAM: 'this', TOKEN_INDEXES: [0]},
            {NGRAM: 'this is', TOKEN_INDEXES: [0, 1]},
            {NGRAM: 'this is a', TOKEN_INDEXES: [0, 1, 2]},
            {NGRAM: 'this is a simple', TOKEN_INDEXES: [0, 1, 2, 3]},
            {NGRAM: 'this is a simple sentence',
             TOKEN_INDEXES: [0, 1, 2, 3, 4]},
            {NGRAM: 'is', TOKEN_INDEXES: [1]},
            {NGRAM: 'is a', TOKEN_INDEXES: [1, 2]},
            {NGRAM: 'is a simple', TOKEN_INDEXES: [1, 2, 3]},
            {NGRAM: 'is a simple sentence', TOKEN_INDEXES: [1, 2, 3, 4]},
            {NGRAM: 'a', TOKEN_INDEXES: [2]},
            {NGRAM: 'a simple', TOKEN_INDEXES: [2, 3]},
            {NGRAM: 'a simple sentence', TOKEN_INDEXES: [2, 3, 4]},
            {NGRAM: 'simple', TOKEN_INDEXES: [3]},
            {NGRAM: 'simple sentence', TOKEN_INDEXES: [3, 4]},
            {NGRAM: 'sentence', TOKEN_INDEXES: [4]}
        ]

        self.assertListEqual(expected_ngrams, ngrams)
