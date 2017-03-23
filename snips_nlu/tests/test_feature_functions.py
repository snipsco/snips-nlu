import re
import unittest

from snips_nlu.slot_filler.feature_functions import (
    char_range_to_token_range, get_regex_match_fn, get_prefix_fn,
    get_suffix_fn, get_ngram_fn, create_feature_function, TOKEN_NAME,
    BaseFeatureFunction, get_token_is_in)


class TestFeatureFunctions(unittest.TestCase):
    def test_ngrams(self):
        # Given
        tokens = ["I", "love", "house", "music"]
        ngrams = {
            1: ["i", "love", "house", "music"],
            2: ["i love", "love house", "house music", None],
            3: ["i love house", "love house music", None, None]
        }

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n)
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    def test_ngrams_with_rare_word(self):
        # Given
        tokens = ["I", "love", "house", "music"]
        ngrams = {
            1: ["i", "love", "rare_word", "music"],
            2: ["i love", "love rare_word", "rare_word music", None],
            3: ["i love rare_word", "love rare_word music", None, None]
        }
        common_words = {"i", "love", "music"}

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n, common_words)
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    # def test_shape_ngrams(self):
    #     assert False

    def test_prefix(self):
        # Given
        tokens = ["AbCde"]
        token = tokens[0]
        expected_prefixes = ["a", "ab", "abc", "abcd", "abcde", None]

        for i in xrange(1, len(token) + 2):
            prefix_fn = get_prefix_fn(i)
            # When
            prefix = prefix_fn.function(tokens, 0)
            # Then
            self.assertEqual(prefix, expected_prefixes[i - 1])

    def test_suffix(self):
        # Given
        tokens = ["AbCde"]
        token = tokens[0]
        expected_suffixes = ["e", "de", "cde", "bcde", "abcde", None]

        for i in xrange(1, len(token) + 2):
            suffix_fn = get_suffix_fn(i)
            # When
            prefix = suffix_fn.function(tokens, 0)
            # Then
            self.assertEqual(prefix, expected_suffixes[i - 1])

    def test_gazetteer(self):
        # Given
        terms = ["bird | rat", "dog pig", "cat"]
        pattern = r"|".join(re.escape(t) for t in terms)
        regex = re.compile(pattern, re.IGNORECASE)

        texts = {
            "there is nothing here": [None, None, None, None],
            "there's a bird | rat here": [None, None, "B-animal", "I-animal",
                                          "I-animal", None],
            "I'm a cat": [None, None, "B-animal"]
        }

        # When
        feature_fn = get_regex_match_fn(regex, "animal", use_bilou=False)

        # Then
        for text, features in texts.iteritems():
            tokens = text.split()
            self.assertEqual(
                features, [feature_fn.function(tokens, i)
                           for i in xrange(len(tokens))])

    def test_gazetteer_with_bilou(self):
        # Given
        terms = ["bird | rat", "dog pig", "cat"]
        pattern = r"|".join(re.escape(t) for t in terms)
        regex = re.compile(pattern, re.IGNORECASE)

        texts = {
            "there is nothing here": [None, None, None, None],
            "there's a bird | rat here": [None, None, "B-animal", "I-animal",
                                          "L-animal", None],
            "I'm a cat": [None, None, "U-animal"]
        }

        # When
        feature_fn = get_regex_match_fn(regex, "animal", use_bilou=True)

        # Then
        for text, features in texts.iteritems():
            tokens = text.split()
            self.assertEqual(features,
                             [feature_fn.function(tokens, i)
                              for i in xrange(len(tokens))])

    def test_token_is_in(self):
        # Given
        collection = {"bIrd"}
        tokens = ["i", "m", "a", "bird"]
        expected_features = ["0", "0", "0", "1"]
        # When
        feature_fn = get_token_is_in(collection, "animal")

        # Then
        self.assertEqual(expected_features,
                         [feature_fn.function(tokens, i)
                          for i in xrange(len(tokens))])

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
            name, lambda tokens, token_index: token_index + 1)

        tokens = ["a", "b", "c"]
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


if __name__ == '__main__':
    unittest.main()
