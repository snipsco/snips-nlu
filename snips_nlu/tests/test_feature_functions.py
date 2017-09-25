# coding=utf-8
from __future__ import unicode_literals

import unittest

import numpy as np
from mock import patch

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.constants import AUTOMATICALLY_EXTENSIBLE, USE_SYNONYMS, \
    SYNONYMS, DATA, VALUE, MATCH_RANGE, ENTITY
from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_utils import TaggingScheme, UNIT_PREFIX, \
    LAST_PREFIX, BEGINNING_PREFIX, INSIDE_PREFIX
from snips_nlu.slot_filler.feature_functions import (
    get_prefix_fn, get_suffix_fn, get_ngram_fn,
    create_feature_function, TOKEN_NAME, BaseFeatureFunction,
    get_token_is_in_fn, get_built_in_annotation_fn, crf_features,
    get_is_in_gazetteer_fn, get_length_fn)
from snips_nlu.tokenization import tokenize


class TestFeatureFunctions(unittest.TestCase):
    def test_ngrams(self):
        # Given
        language = Language.EN
        tokens = tokenize("I love house music", language)
        ngrams = {
            1: ["i", "love", "house", "music"],
            2: ["i love", "love house", "house music", None],
            3: ["i love house", "love house music", None, None]
        }

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n, use_stemming=False,
                                     language_code=language.iso_code)
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    def test_chinese_ngrams(self):
        # Given
        language = Language.ZH
        tokens = tokenize("我喜欢流行音乐", language)
        ngrams = {
            1: ["我", "喜欢", "流行音乐"],
            2: ["我喜欢", "喜欢流行音乐", None],
            3: ["我喜欢流行音乐", None, None]
        }

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n, use_stemming=False,
                                     language_code=language.iso_code)
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    @patch('snips_nlu.slot_filler.feature_functions.get_gazetteer')
    def test_ngrams_with_rare_word(self, mocked_get_gazetteer):
        # Given
        mocked_gazetteer = {"i", "love", "music"}

        mocked_get_gazetteer.return_value = mocked_gazetteer
        language = Language.EN
        tokens = tokenize("I love house Müsic", language)
        ngrams = {
            1: ["i", "love", "rare_word", "music"],
            2: ["i love", "love rare_word", "rare_word music", None],
            3: ["i love rare_word", "love rare_word music", None, None]
        }

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n, use_stemming=False,
                                     language_code=language.iso_code,
                                     common_words_gazetteer_name='common')
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    @patch('snips_nlu.slot_filler.feature_functions.get_gazetteer')
    def test_chinese_ngrams_with_rare_word(self, mocked_get_gazetteer):
        # Given
        mocked_gazetteer = {"我", "喜欢"}

        mocked_get_gazetteer.return_value = mocked_gazetteer
        language = Language.ZH
        tokens = tokenize("我喜欢流行音乐", language)
        ngrams = {
            1: ["我", "喜欢", "rare_word"],
            2: ["我喜欢", "喜欢rare_word", None],
            3: ["我喜欢rare_word", None, None]
        }

        for n, expected_features in ngrams.iteritems():
            ngrams_fn = get_ngram_fn(n, use_stemming=False,
                                     language_code=language.iso_code,
                                     common_words_gazetteer_name='common')
            # When
            features = [ngrams_fn.function(tokens, i)
                        for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_features, features)

    def test_prefix(self):
        # Given
        language = Language.EN
        tokens = tokenize("AbCde", language)
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
        language = Language.EN
        tokens = tokenize("AbCde", language)
        token = tokens[0]
        expected_suffixes = ["e", "de", "cde", "bcde", "abcde", None]

        for i in xrange(1, len(token.value) + 2):
            suffix_fn = get_suffix_fn(i)
            # When
            prefix = suffix_fn.function(tokens, 0)
            # Then
            self.assertEqual(prefix, expected_suffixes[i - 1])

    def test_length(self):
        # Given
        language = Language.EN
        tokens = tokenize("i'm here dude", language)

        # When
        fn = get_length_fn().function
        tokens_length = [fn(tokens, i) for i in xrange(len(tokens))]

        # Then
        expected_tokens_lengths = [1, 1, 4, 4]
        self.assertSequenceEqual(tokens_length, expected_tokens_lengths)

    def test_token_is_in(self):
        # Given
        language = Language.EN
        collection = {"bird", "blue bird"}
        tokens = tokenize("i m a Blue bÏrd", language)
        expected_features = [None, None, None, BEGINNING_PREFIX, LAST_PREFIX]
        # When
        scheme_code = TaggingScheme.BILOU.value
        feature_fn = get_token_is_in_fn(collection, "animal",
                                        use_stemming=False,
                                        tagging_scheme_code=scheme_code,
                                        language_code='en')

        # Then
        self.assertEqual(expected_features,
                         [feature_fn.function(tokens, i)
                          for i in xrange(len(tokens))])

    @patch('snips_nlu.preprocessing.stem')
    def test_token_is_in_with_stemming(self, mocked_stem):
        # Given
        def stem_mocked(string, _):
            if string.startswith("bird"):
                return "bird_stemmed"
            else:
                return string

        mocked_stem.side_effect = stem_mocked
        language = Language.EN
        collection = {"bird_stemmed", "blue bird_stemmed"}
        tokens = tokenize("i m a Blue bÏrdy", language)
        for token in tokens:
            token.stem = stem_mocked(token.normalized_value, language)
        expected_features = [None, None, None, BEGINNING_PREFIX, LAST_PREFIX]
        # When
        scheme_code = TaggingScheme.BILOU.value
        feature_fn = get_token_is_in_fn(collection, "animal",
                                        use_stemming=True,
                                        tagging_scheme_code=scheme_code,
                                        language_code=language.iso_code)

        # Then
        self.assertEqual(expected_features,
                         [feature_fn.function(tokens, i)
                          for i in xrange(len(tokens))])

    def test_chinese_token_is_in(self):
        # Given
        language = Language.ZH
        collection = {"花", "红花"}
        tokens = tokenize("这是一朵红花", language)
        expected_features = [None, None, UNIT_PREFIX]
        # When
        scheme_code = TaggingScheme.BILOU.value
        feature_fn = get_token_is_in_fn(collection, "flower",
                                        use_stemming=False,
                                        tagging_scheme_code=scheme_code,
                                        language_code='en')

        # Then
        self.assertEqual(expected_features,
                         [feature_fn.function(tokens, i)
                          for i in xrange(len(tokens))])

    @patch('snips_nlu.slot_filler.feature_functions.get_gazetteer')
    def test_is_in_gazetteer(self, mocked_get_gazetteer):
        # Given
        gazetteer = {"bird", "eagle", "blue bird"}
        mocked_get_gazetteer.side_effect = lambda language, name: gazetteer
        text = "This is a Blue bÏrd flying next to an eagle"
        language = Language.EN
        tokens = tokenize(text, language=language)
        feature_fn = get_is_in_gazetteer_fn("bird_gazetteer",
                                            language.iso_code,
                                            TaggingScheme.BILOU.value,
                                            use_stemming=False)

        # When
        features = [feature_fn.function(tokens, i) for i in
                    xrange(len(tokens))]

        # Then
        expected_features = [None, None, None, BEGINNING_PREFIX, LAST_PREFIX,
                             None, None, None, None, UNIT_PREFIX]
        self.assertListEqual(features, expected_features)

    @patch('snips_nlu.slot_filler.feature_functions.get_gazetteer')
    def test_chinese_is_in_gazetteer(self, mocked_get_gazetteer):
        # Given
        gazetteer = {"花", "红花", "蓝花"}
        mocked_get_gazetteer.side_effect = lambda language, name: gazetteer
        text = "红花旁边有一朵蓝花"
        language = Language.ZH
        tokens = tokenize(text, language=language)
        feature_fn = get_is_in_gazetteer_fn("bird_gazetteer",
                                            language.iso_code,
                                            TaggingScheme.BILOU.value,
                                            use_stemming=False)

        # When
        features = [feature_fn.function(tokens, i) for i in
                    xrange(len(tokens))]

        # Then
        expected_features = [UNIT_PREFIX, None, None, None, UNIT_PREFIX]
        self.assertListEqual(features, expected_features)

    @patch('snips_nlu.slot_filler.feature_functions.get_builtin_entities')
    def test_get_built_in_annotation_fn(self, mocked_get_builtin_entities):
        # Given
        language = Language.EN
        input_text = u"i ll be there tomorrow at noon   is that ok"

        mocked_built_in_entities = [
            {
                MATCH_RANGE: (14, 30),
                VALUE: u"tomorrow at noon",
                ENTITY: BuiltInEntity.DATETIME
            }
        ]

        mocked_get_builtin_entities.return_value = mocked_built_in_entities
        tokens = tokenize(input_text, language)
        feature_fn = get_built_in_annotation_fn(BuiltInEntity.DATETIME.label,
                                                Language.EN.iso_code,
                                                TaggingScheme.BILOU.value)
        expected_features = [None, None, None, None, BEGINNING_PREFIX,
                             INSIDE_PREFIX, LAST_PREFIX, None, None, None]

        # When
        features = [feature_fn.function(tokens, i)
                    for i in xrange(len(tokens))]

        # Then
        self.assertEqual(features, expected_features)

    def test_create_feature_function(self):
        # Given
        language = Language.EN
        name = "position"
        base_feature_function = BaseFeatureFunction(
            name, lambda _, token_index: token_index + 1)

        tokens = tokenize("a b c", language)
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

    def test_crf_features(self):
        # Given
        language = Language.EN
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
        features_signatures = crf_features(intent_entities=intent_entities,
                                           language=language)

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
                    'tokens_collection': collection_1,
                    'collection_name': 'dummy_entity_1',
                    'use_stemming': True,
                    'language_code': 'en',
                    'tagging_scheme_code': TaggingScheme.BILOU.value
                },
                'factory_name': 'get_token_is_in_fn',
                'offsets': (-2, -1, 0)
            },
            {
                'args': {
                    'tokens_collection': collection_2,
                    'collection_name': 'dummy_entity_2',
                    'use_stemming': True,
                    'language_code': 'en',
                    'tagging_scheme_code': TaggingScheme.BILOU.value
                },
                'factory_name': 'get_token_is_in_fn',
                'offsets': (-2, -1, 0)
            }
        ]
        for signature in expected_signatures:
            self.assertIn(signature, features_signatures)


if __name__ == '__main__':
    unittest.main()
