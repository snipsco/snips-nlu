# coding=utf-8
from __future__ import unicode_literals

from copy import deepcopy

from mock import MagicMock, patch

from snips_nlu.builtin_entities import BuiltinEntityParser
from snips_nlu.constants import LANGUAGE, LANGUAGE_EN, SNIPS_DATETIME, \
    SNIPS_NUMBER
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.preprocessing import tokenize
from snips_nlu.slot_filler.crf_utils import (
    BEGINNING_PREFIX, INSIDE_PREFIX, LAST_PREFIX, TaggingScheme, UNIT_PREFIX)
from snips_nlu.slot_filler.feature import Feature, TOKEN_NAME
from snips_nlu.slot_filler.feature_factory import (
    BuiltinEntityMatchFactory, EntityMatchFactory, IsDigitFactory,
    IsFirstFactory, IsLastFactory, LengthFactory, NgramFactory, PrefixFactory,
    ShapeNgramFactory, SingleFeatureFactory, SuffixFactory, WordClusterFactory,
    get_feature_factory)
from snips_nlu.tests.utils import SAMPLE_DATASET, SnipsTest


class TestCRFFeatures(SnipsTest):
    def test_feature_should_work(self):
        # Given
        def fn(tokens, token_index):
            value = tokens[token_index].value
            return "%s_%s" % (value, len(value))

        cache = [{TOKEN_NAME: token} for token in
                 tokenize("hello beautiful world", LANGUAGE_EN)]
        feature = Feature("test_feature", fn)

        # When
        res = feature.compute(1, cache)

        # Then
        self.assertEqual(res, "beautiful_9")

    def test_feature_should_work_with_offset(self):
        # Given
        def fn(tokens, token_index):
            value = tokens[token_index].value
            return "%s_%s" % (value, len(value))

        cache = [{TOKEN_NAME: token} for token in
                 tokenize("hello beautiful world", LANGUAGE_EN)]
        feature = Feature("test_feature", fn, offset=1)

        # When
        res = feature.compute(1, cache)

        # Then
        self.assertEqual(res, "world_5")

    def test_feature_should_work_with_cache(self):
        # Given
        def fn(tokens, token_index):
            value = tokens[token_index].value
            return "%s_%s" % (value, len(value))

        mocked_fn = MagicMock(side_effect=fn)

        cache = [{TOKEN_NAME: token} for token in
                 tokenize("hello beautiful world", LANGUAGE_EN)]
        feature = Feature("test_feature", mocked_fn, offset=0)
        feature.compute(2, cache)
        feature1 = Feature("test_feature", mocked_fn, offset=1)
        feature2 = Feature("test_feature", mocked_fn, offset=2)

        # When
        res1 = feature1.compute(1, cache)
        res1_bis = feature1.compute(0, cache)
        res2 = feature2.compute(0, cache)

        # Then
        self.assertEqual(res1, "world_5")
        self.assertEqual(res1_bis, "beautiful_9")
        self.assertEqual(res2, "world_5")
        self.assertEqual(mocked_fn.call_count, 2)

    def test_single_feature_factory(self):
        # Given
        class TestSingleFeatureFactory(SingleFeatureFactory):
            def compute_feature(self, tokens, token_index):
                value = tokens[token_index].value
                return "%s_%s" % (value, len(value))

        config = {
            "factory_name": "test_factory",
            "args": {},
            "offsets": [0, 1]
        }
        factory = TestSingleFeatureFactory(config)
        factory.fit(None, None)
        features = factory.build_features()
        cache = [{TOKEN_NAME: token} for token in
                 tokenize("hello beautiful world", LANGUAGE_EN)]

        # When
        res_0 = features[0].compute(0, cache)
        res_1 = features[1].compute(0, cache)

        # Then
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].name, "test_factory")
        self.assertEqual(features[1].name, "test_factory[+1]")
        self.assertEqual(res_0, "hello_5")
        self.assertEqual(res_1, "beautiful_9")

    def test_is_digit_factory(self):
        # Given
        config = {
            "factory_name": "is_digit",
            "args": {},
            "offsets": [0]
        }
        tokens = tokenize("hello 1 world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        factory.fit(None, None)
        features = factory.build_features()

        # When
        res1 = features[0].compute(0, cache)
        res2 = features[0].compute(1, cache)

        # Then
        self.assertIsInstance(factory, IsDigitFactory)
        self.assertEqual(features[0].base_name, "is_digit")
        self.assertEqual(res1, None)
        self.assertEqual(res2, "1")

    def test_is_first_factory(self):
        # Given
        config = {
            "factory_name": "is_first",
            "args": {},
            "offsets": [0]
        }
        tokens = tokenize("hello beautiful world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        factory.fit(None, None)
        features = factory.build_features()

        # When
        res1 = features[0].compute(0, cache)
        res2 = features[0].compute(1, cache)

        # Then
        self.assertIsInstance(factory, IsFirstFactory)
        self.assertEqual(features[0].base_name, "is_first")
        self.assertEqual(res1, "1")
        self.assertEqual(res2, None)

    def test_is_last_factory(self):
        # Given
        config = {
            "factory_name": "is_last",
            "args": {},
            "offsets": [0]
        }
        tokens = tokenize("hello beautiful world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        factory.fit(None, None)
        features = factory.build_features()

        # When
        res1 = features[0].compute(0, cache)
        res2 = features[0].compute(2, cache)

        # Then
        self.assertIsInstance(factory, IsLastFactory)
        self.assertEqual(features[0].base_name, "is_last")
        self.assertEqual(res1, None)
        self.assertEqual(res2, "1")

    def test_prefix_factory(self):
        # Given
        config = {
            "factory_name": "prefix",
            "args": {
                "prefix_size": 2
            },
            "offsets": [0]
        }
        tokens = tokenize("hello beautiful world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        factory.fit(None, None)
        features = factory.build_features()

        # When
        res = features[0].compute(1, cache)

        # Then
        self.assertIsInstance(factory, PrefixFactory)
        self.assertEqual(features[0].base_name, "prefix_2")
        self.assertEqual(res, "be")

    def test_suffix_factory(self):
        # Given
        config = {
            "factory_name": "suffix",
            "args": {
                "suffix_size": 2
            },
            "offsets": [0]
        }
        tokens = tokenize("hello beautiful world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        factory.fit(None, None)
        features = factory.build_features()

        # When
        res = features[0].compute(1, cache)

        # Then
        self.assertIsInstance(factory, SuffixFactory)
        self.assertEqual(features[0].base_name, "suffix_2")
        self.assertEqual(res, "ul")

    def test_length_factory(self):
        # Given
        config = {
            "factory_name": "length",
            "args": {},
            "offsets": [0]
        }
        tokens = tokenize("hello beautiful world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        factory.fit(None, None)
        features = factory.build_features()

        # When
        res = features[0].compute(2, cache)

        # Then
        self.assertIsInstance(factory, LengthFactory)
        self.assertEqual(features[0].base_name, "length")
        self.assertEqual(res, "5")

    def test_ngram_factory(self):
        # Given
        config = {
            "factory_name": "ngram",
            "args": {
                "n": 2,
                "use_stemming": False,
                "common_words_gazetteer_name": None
            },
            "offsets": [0]
        }
        tokens = tokenize("hello beautiful world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res = features[0].compute(0, cache)

        # Then
        self.assertIsInstance(factory, NgramFactory)
        self.assertEqual(features[0].base_name, "ngram_2")
        self.assertEqual(res, "hello beautiful")

    @patch("snips_nlu.slot_filler.feature_factory.get_gazetteer")
    def test_ngram_factory_with_gazetteer(self, mock_get_gazetteer):
        # Given
        config = {
            "factory_name": "ngram",
            "args": {
                "n": 2,
                "use_stemming": False,
                "common_words_gazetteer_name": "mocked_gazetteer"
            },
            "offsets": [0]
        }

        mock_get_gazetteer.return_value = {"hello", "beautiful", "world"}
        tokens = tokenize("hello beautiful foobar world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res = features[0].compute(1, cache)

        # Then
        self.assertIsInstance(factory, NgramFactory)
        self.assertEqual(features[0].base_name, "ngram_2")
        self.assertEqual(res, "beautiful rare_word")

    def test_shape_ngram_factory(self):
        # Given
        config = {
            "factory_name": "shape_ngram",
            "args": {
                "n": 3,
            },
            "offsets": [0]
        }

        tokens = tokenize("hello Beautiful foObar world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res = features[0].compute(1, cache)

        # Then
        self.assertIsInstance(factory, ShapeNgramFactory)
        self.assertEqual(features[0].base_name, "shape_ngram_3")
        self.assertEqual(res, "Xxx xX xxx")

    @patch("snips_nlu.slot_filler.feature_factory.get_word_clusters")
    def test_word_cluster_factory(self, mock_get_word_clusters):
        # Given
        def mocked_get_word_clusters(language):
            if language == LANGUAGE_EN:
                return {
                    "mocked_cluster": {
                        "word1": "00",
                        "word2": "11"
                    }
                }
            return dict()

        mock_get_word_clusters.side_effect = mocked_get_word_clusters

        config = {
            "factory_name": "word_cluster",
            "args": {
                "cluster_name": "mocked_cluster",
                "use_stemming": False
            },
            "offsets": [0]
        }

        tokens = tokenize("hello word1 word2", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res0 = features[0].compute(0, cache)
        res1 = features[0].compute(1, cache)
        res2 = features[0].compute(2, cache)

        # Then
        self.assertIsInstance(factory, WordClusterFactory)
        self.assertEqual(features[0].base_name, "word_cluster_mocked_cluster")
        self.assertEqual(res0, None)
        self.assertEqual(res1, "00")
        self.assertEqual(res2, "11")

    def test_entity_match_factory(self):
        # Given
        config = {
            "factory_name": "entity_match",
            "args": {
                "tagging_scheme_code": TaggingScheme.BILOU.value,
                "use_stemming": False
            },
            "offsets": [0]
        }

        tokens = tokenize("2 dummy a and dummy_c", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        dataset = deepcopy(SAMPLE_DATASET)
        dataset = validate_and_format_dataset(dataset)
        factory.fit(dataset, "dummy_intent_1")

        # When
        features = factory.build_features()
        features = sorted(features, key=lambda f: f.base_name)
        res0 = features[0].compute(0, cache)
        res1 = features[0].compute(1, cache)
        res2 = features[0].compute(2, cache)
        res3 = features[0].compute(3, cache)
        res4 = features[0].compute(4, cache)

        res5 = features[1].compute(0, cache)
        res6 = features[1].compute(1, cache)
        res7 = features[1].compute(2, cache)
        res8 = features[1].compute(3, cache)
        res9 = features[1].compute(4, cache)

        # Then
        self.assertIsInstance(factory, EntityMatchFactory)
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].base_name, "entity_match_dummy_entity_1")
        self.assertEqual(features[1].base_name, "entity_match_dummy_entity_2")
        self.assertEqual(res0, BEGINNING_PREFIX)
        self.assertEqual(res1, INSIDE_PREFIX)
        self.assertEqual(res2, LAST_PREFIX)
        self.assertEqual(res3, None)
        self.assertEqual(res4, None)

        self.assertEqual(res5, None)
        self.assertEqual(res6, None)
        self.assertEqual(res7, None)
        self.assertEqual(res8, None)
        self.assertEqual(res9, UNIT_PREFIX)

    def test_builtin_entity_match_factory(self):
        # Given
        def mock_builtin_entity_scope(dataset, intent):
            if dataset[LANGUAGE] == LANGUAGE_EN:
                return {SNIPS_NUMBER, SNIPS_DATETIME}
            return []

        config = {
            "factory_name": "builtin_entity_match",
            "args": {
                "tagging_scheme_code": TaggingScheme.BILOU.value,
            },
            "offsets": [0]
        }

        tokens = tokenize("one tea tomorrow at 2pm", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = get_feature_factory(config)
        factory._get_builtin_entity_scope = mock_builtin_entity_scope
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)

        # When
        features = factory.build_features(BuiltinEntityParser("en", None))
        features = sorted(features, key=lambda f: f.base_name)
        res0 = features[0].compute(0, cache)
        res1 = features[0].compute(1, cache)
        res2 = features[0].compute(2, cache)
        res3 = features[0].compute(3, cache)
        res4 = features[0].compute(4, cache)

        res5 = features[1].compute(0, cache)
        res6 = features[1].compute(1, cache)
        res7 = features[1].compute(2, cache)
        res8 = features[1].compute(3, cache)
        res9 = features[1].compute(4, cache)

        # Then
        self.assertIsInstance(factory, BuiltinEntityMatchFactory)
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].base_name,
                         "builtin_entity_match_snips/datetime")
        self.assertEqual(features[1].base_name,
                         "builtin_entity_match_snips/number")
        self.assertEqual(res0, UNIT_PREFIX)
        self.assertEqual(res1, None)
        self.assertEqual(res2, BEGINNING_PREFIX)
        self.assertEqual(res3, INSIDE_PREFIX)
        self.assertEqual(res4, LAST_PREFIX)

        self.assertEqual(res5, UNIT_PREFIX)
        self.assertEqual(res6, None)
        self.assertEqual(res7, None)
        self.assertEqual(res8, None)
        self.assertEqual(res9, None)
