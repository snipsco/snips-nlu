# coding=utf-8
from __future__ import unicode_literals

import unittest

from mock import patch

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.config import CRFSlotFillerConfig
from snips_nlu.constants import AUTOMATICALLY_EXTENSIBLE, USE_SYNONYMS, \
    SYNONYMS, DATA, VALUE, MATCH_RANGE, ENTITY
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_utils import TaggingScheme, LAST_PREFIX, \
    BEGINNING_PREFIX, INSIDE_PREFIX
from snips_nlu.slot_filler.feature_functions import (
    get_prefix_fn, get_suffix_fn, get_ngram_fn, TOKEN_NAME, Feature,
    get_token_is_in_fn, get_built_in_annotation_fn, crf_features,
    get_length_fn)
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
                                        language_code=language.iso_code)

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

    def test_offset_feature(self):
        # Given
        language = Language.EN
        name = "position"
        base_feature_function = Feature(
            name, lambda _, token_index: token_index + 1)

        tokens = tokenize("a b c", language)
        expected_features = {
            0: ("position", [1, 2, 3]),
            -1: ("position[-1]", [None, 1, 2]),
            1: ("position[+1]", [2, 3, None]),
            2: ("position[+2]", [3, None, None])
        }
        cache = [{TOKEN_NAME: t for t in tokens} for _ in xrange(len(tokens))]
        for offset, (expected_name, expected_values) in \
                expected_features.iteritems():
            offset_feature = base_feature_function.get_offset_feature(offset)
            # When
            feature_values = [offset_feature.compute(i, cache)
                              for i in xrange(len(tokens))]
            # Then
            self.assertEqual(expected_name, offset_feature.name)
            self.assertEqual(expected_values, feature_values)

    def test_crf_features(self):
        # Given
        language = Language.EN
        dataset = {
            "snips_nlu_version": "1.0.0",
            "language": "en",
            "intents": {
                "dummy_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "hello",
                                    "entity": "dummy_entity_1",
                                    "slot_name": "dummy_slot_1"
                                },
                                {
                                    "text": "there",
                                    "entity": "dummy_entity_2",
                                    "slot_name": "dummy_slot_2"
                                },
                            ]
                        }
                    ]
                }
            },
            "entities": {
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
        }
        dataset = validate_and_format_dataset(dataset)

        collection_1 = {
            'dummy_a': 'dummy_a',
            'dummya': 'dummy_a',
            'dummy_a_bis': 'dummy_a',
            'dummyabis': 'dummy_a',
            'dummya_bis': 'dummy_a',
            'dummy_abis': 'dummy_a',
            'dummy_b': 'dummy_b',
            'dummyb': 'dummy_b',
            'dummy_b_bis': 'dummy_b',
            'dummybbis': 'dummy_b',
            'dummy_bbis': 'dummy_b',
            'dummyb_bis': 'dummy_b',
            'hello': 'hello'
        }

        collection_2 = {
            'dummy_c': 'dummy_c',
            'dummyc': 'dummyc',
            'there': 'there'
        }

        # When
        features_config = CRFSlotFillerConfig()
        features_signatures = crf_features(
            dataset, "dummy_1", language=language,
            crf_features_config=features_config)

        # Then
        for signature in features_signatures:
            # sort collections in order to make testing easier
            if 'tokens_collection' in signature['args']:
                signature['args']['tokens_collection'] = sorted(
                    signature['args']['tokens_collection'])

        expected_signatures = [
            {
                'args': {
                    'tokens_collection':
                        sorted(list(set(collection_1.keys()))),
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
                    'tokens_collection':
                        sorted(list(set(collection_2.keys()))),
                    'collection_name': 'dummy_entity_2',
                    'use_stemming': True,
                    'language_code': 'en',
                    'tagging_scheme_code': TaggingScheme.BILOU.value
                },
                'factory_name': 'get_token_is_in_fn',
                'offsets': (-2, -1, 0)
            }
        ]
        for expected_signature in expected_signatures:
            self.assertIn(expected_signature, features_signatures)


if __name__ == '__main__':
    unittest.main()
