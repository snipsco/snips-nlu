import re
import unittest

import numpy as np


from snips_nlu.built_in_entities import BuiltInEntity
from snips_nlu.constants import AUTOMATICALLY_EXTENSIBLE, USE_SYNONYMS, \
    SYNONYMS, DATA, VALUE
from snips_nlu.languages import Language
from snips_nlu.slot_filler.feature_functions import (
    char_range_to_token_range, get_regex_match_fn, get_prefix_fn,
    get_suffix_fn, get_ngram_fn, create_feature_function, TOKEN_NAME,
    BaseFeatureFunction, get_token_is_in, get_built_in_annotation_fn,
    crf_features)
from snips_nlu.tokenization import tokenize


class TestFeatureFunctions(unittest.TestCase):
    def test_ngrams(self):
        # Given
        tokens = tokenize("I love house music")
        ngrams = {
            1: ["i", "love", "house", "music"],
            2: ["i love", "love house", "house music", None],
            3: ["i love house", "love house music", None, None]
        }

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n, use_stemming=False)
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    def test_ngrams_with_rare_word(self):
        # Given
        tokens = tokenize("I love house music")
        ngrams = {
            1: ["i", "love", "rare_word", "music"],
            2: ["i love", "love rare_word", "rare_word music", None],
            3: ["i love rare_word", "love rare_word music", None, None]
        }
        common_words = {"i", "love", "music"}

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n, use_stemming=False,
                                     common_words=common_words)
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    # def test_shape_ngrams(self):
    #     assert False

    def test_prefix(self):
        # Given
        tokens = tokenize("AbCde")
        token = tokens[0]
        expected_prefixes = ["a", "ab", "abc", "abcd", "abcde", None]

        for i in xrange(1, len(token.value) + 2):
            prefix_fn = get_prefix_fn(i)
            # When
            prefix = prefix_fn.function(tokens, 0)
            # Then
            self.assertEqual(prefix, expected_prefixes[i - 1])

    def test_suffix(self):
        # Given
        tokens = tokenize("AbCde")
        token = tokens[0]
        expected_suffixes = ["e", "de", "cde", "bcde", "abcde", None]

        for i in xrange(1, len(token.value) + 2):
            suffix_fn = get_suffix_fn(i)
            # When
            prefix = suffix_fn.function(tokens, 0)
            # Then
            self.assertEqual(prefix, expected_suffixes[i - 1])

    def test_regex_match(self):
        # Given
        terms = ["bird cow rat", "dog pig", "cat"]
        pattern = r"|".join(re.escape(t) for t in terms)
        regex = re.compile(pattern, re.IGNORECASE)

        texts = {
            "there is nothing here": [None, None, None, None],
            "there s a bird cow rat here": [None, None, None, "B-animal",
                                            "I-animal", "I-animal", None],
            "I m a cat": [None, None, None, "B-animal"]
        }

        # When
        feature_fn = get_regex_match_fn(regex, "animal", use_bilou=False)

        # Then
        for text, features in texts.iteritems():
            tokens = tokenize(text)
            self.assertEqual(features,
                             [feature_fn.function(tokens, i)
                              for i in xrange(len(tokens))])

    def test_regex_match_with_bilou(self):
        # Given
        terms = ["bird cow rat", "dog pig", "cat"]
        pattern = r"|".join(re.escape(t) for t in terms)
        regex = re.compile(pattern, re.IGNORECASE)

        texts = {
            "there is nothing here": [None, None, None, None],
            "there s a bird cow rat here": [None, None, None, "B-animal",
                                            "I-animal", "L-animal", None],
            "I m a cat": [None, None, None, "U-animal"]
        }

        # When
        feature_fn = get_regex_match_fn(regex, "animal", use_bilou=True)

        # Then
        for text, features in texts.iteritems():
            tokens = tokenize(text)
            self.assertEqual(features,
                             [feature_fn.function(tokens, i)
                              for i in xrange(len(tokens))])

    def test_token_is_in(self):
        # Given
        collection = {"bIrd"}
        tokens = tokenize("i m a bird")
        expected_features = [None, None, None, "1"]
        # When
        feature_fn = get_token_is_in(collection, "animal", use_stemming=False)

        # Then
        self.assertEqual(expected_features,
                         [feature_fn.function(tokens, i)
                          for i in xrange(len(tokens))])

    def test_get_built_in_annotation_fn(self):
        # Given
        language = "en"
        language = Language.from_iso_code(language)
        text = "i ll be there tomorrow at noon   is that ok?"
        tokens = tokenize(text)
        built_in = BuiltInEntity.DATETIME
        feature_fn = get_built_in_annotation_fn(built_in.label,
                                                language.iso_code)
        expected_features = [None, None, None, None, "1", "1", "1", None, None,
                             None]

        # When
        features = [feature_fn.function(tokens, i)
                    for i in xrange(len(tokens))]

        # Then
        self.assertEqual(features, expected_features)

    def test_char_range_to_token_range(self):
        # Given
        text = "I'm here for eating"
        tokens = text.split()
        char_to_token_range = {
            (0, 3): (0, 1),
            (5, 6): None,
            (4, 8): (1, 2),
            (5, 11): None,
            (4, 12): (1, 3),
            (13, 19): (3, 4)
        }
        # When/Then
        for char_range, token_range in char_to_token_range.iteritems():
            self.assertEqual(char_range_to_token_range(char_range, tokens),
                             token_range)

    def test_create_feature_function(self):
        # Given
        name = "position"
        base_feature_function = BaseFeatureFunction(
            name, lambda _, token_index: token_index + 1)

        tokens = tokenize("a b c")
        expected_features = {
            0: ("position", [1, 2, 3]),
            -1: ("position[-1]", [None, 1, 2]),
            1: ("position[+1]", [2, 3, None]),
            2: ("position[+2]", [3, None, None])
        }
        cache = [{TOKEN_NAME: t for t in tokens} for _ in xrange(len(tokens))]
        for offset, expected in expected_features.iteritems():
            feature_name, feature_function = create_feature_function(
                base_feature_function, offset)
            expected_name, expected_feats = expected
            # When
            feats = [feature_function(i, cache) for i in xrange(len(tokens))]
            # Then
            self.assertEqual(feature_name, expected_name)
            self.assertEqual(feats, expected_feats)

    # @patch("snips_nlu.slot_filler.feature_functions.default_features")
    def test_crf_features(self, ):
        def mocked_default(language, intent_entities, use_stemming,
                           entities_offsets, entity_keep_prob):
            return []

        # mocked_default_features.side_effect = mocked_default

        intent_entities = {
            "dummy_entity_1": {
                AUTOMATICALLY_EXTENSIBLE: False,
                USE_SYNONYMS: True,
                DATA: [
                    {
                        SYNONYMS: [
                            "dummy_a",
                            "dummy_a_bis"
                        ],
                        VALUE: "dummy_a"
                    },
                    {
                        SYNONYMS: [
                            "dummy_b",
                            "dummy_b_bis"
                        ],
                        VALUE: "dummy_b"
                    }
                ]
            },
            "dummy_entity_2": {
                AUTOMATICALLY_EXTENSIBLE: False,
                USE_SYNONYMS: False,
                DATA: [
                    {
                        SYNONYMS: [
                            "dummy_c",
                            "dummy_c_bis"
                        ],
                        VALUE: "dummy_c"
                    }
                ]
            }
        }

        # When
        np.random.seed(42)
        keep_prob = 0.5
        features_signatures = crf_features(
            intent_entities=intent_entities,
            language=Language.EN)

        # Then
        np.random.seed(42)
        collection_1 = ['dummy_a', 'dummy_a_bis', 'dummy_b', 'dummy_b_bis']
        collection_1_size = max(int(keep_prob * len(collection_1)), 1)
        collection_1 = np.random.choice(collection_1, collection_1_size,
                                        replace=False).tolist()
        collection_2 = ['dummy_c']

        expected_signatures = [
            {
                'args': {
                    'collection': collection_1,
                    'collection_name': 'dummy_entity_1',
                    'use_stemming': True
                },
                'factory_name': 'get_token_is_in',
                'module_name': 'snips_nlu.slot_filler.feature_functions',
                'offsets': (-2, -1, 0)
            },
            {
                'args': {
                    'collection': collection_2,
                    'collection_name': 'dummy_entity_2',
                    'use_stemming': True
                },
                'factory_name': 'get_token_is_in',
                'module_name': 'snips_nlu.slot_filler.feature_functions',
                'offsets': (-2, -1, 0)
            }
        ]
        for signature in expected_signatures:
            self.assertIn(signature, features_signatures)


if __name__ == '__main__':
    unittest.main()
