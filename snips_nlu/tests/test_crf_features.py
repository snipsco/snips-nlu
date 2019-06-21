# coding=utf-8
from __future__ import unicode_literals

import io

from mock import MagicMock

from snips_nlu.constants import LANGUAGE, LANGUAGE_EN, SNIPS_DATETIME, \
    SNIPS_NUMBER, STEMS, GAZETTEERS, WORD_CLUSTERS
from snips_nlu.dataset import Dataset
from snips_nlu.entity_parser import BuiltinEntityParser, CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser_usage import \
    CustomEntityParserUsage
from snips_nlu.exceptions import NotRegisteredError
from snips_nlu.preprocessing import tokenize
from snips_nlu.slot_filler.crf_utils import (
    BEGINNING_PREFIX, INSIDE_PREFIX, LAST_PREFIX, TaggingScheme, UNIT_PREFIX)
from snips_nlu.slot_filler.feature import Feature, TOKEN_NAME
from snips_nlu.slot_filler.feature_factory import (
    BuiltinEntityMatchFactory, CustomEntityMatchFactory, IsDigitFactory,
    IsFirstFactory, IsLastFactory, LengthFactory, NgramFactory, PrefixFactory,
    ShapeNgramFactory, SingleFeatureFactory, SuffixFactory, WordClusterFactory,
    CRFFeatureFactory)
from snips_nlu.tests.utils import SnipsTest


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
        @CRFFeatureFactory.register("my_factory", override=True)
        class MySingleFeatureFactory(SingleFeatureFactory):
            def compute_feature(self, tokens, token_index):
                value = tokens[token_index].value
                return "%s_%s" % (value, len(value))

        config = {
            "factory_name": "my_factory",
            "args": {},
            "offsets": [0, 1]
        }
        factory = MySingleFeatureFactory(config)
        factory.fit(None, None)
        features = factory.build_features()
        cache = [{TOKEN_NAME: token} for token in
                 tokenize("hello beautiful world", LANGUAGE_EN)]

        # When
        res_0 = features[0].compute(0, cache)
        res_1 = features[1].compute(0, cache)

        # Then
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].name, "my_factory")
        self.assertEqual(features[1].name, "my_factory[+1]")
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
        factory = CRFFeatureFactory.from_config(config)
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
        factory = CRFFeatureFactory.from_config(config)
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
        factory = CRFFeatureFactory.from_config(config)
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
        factory = CRFFeatureFactory.from_config(config)
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
        factory = CRFFeatureFactory.from_config(config)
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
        factory = CRFFeatureFactory.from_config(config)
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
        factory = CRFFeatureFactory.from_config(config)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res = features[0].compute(0, cache)

        # Then
        self.assertIsInstance(factory, NgramFactory)
        self.assertEqual(features[0].base_name, "ngram_2")
        self.assertEqual(res, "hello beautiful")

    def test_ngram_factory_with_stemming(self):
        # Given
        config = {
            "factory_name": "ngram",
            "args": {
                "n": 2,
                "use_stemming": True,
                "common_words_gazetteer_name": None
            },
            "offsets": [0]
        }
        tokens = tokenize("hello beautiful world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        resources = {STEMS: {"beautiful": "beauty"}}
        factory = CRFFeatureFactory.from_config(config, resources=resources)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res = features[0].compute(0, cache)

        # Then
        self.assertIsInstance(factory, NgramFactory)
        self.assertEqual(features[0].base_name, "ngram_2")
        self.assertEqual(res, "hello beauty")

    def test_ngram_factory_with_gazetteer(self):
        # Given
        config = {
            "factory_name": "ngram",
            "args": {
                "n": 2,
                "use_stemming": False,
                "common_words_gazetteer_name": "my_gazetteer"
            },
            "offsets": [0]
        }
        resources = {
            GAZETTEERS: {
                "my_gazetteer": {"hello", "beautiful", "world"}
            }
        }
        tokens = tokenize("hello beautiful foobar world", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = CRFFeatureFactory.from_config(config, resources=resources)
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
        factory = CRFFeatureFactory.from_config(config)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res = features[0].compute(1, cache)

        # Then
        self.assertIsInstance(factory, ShapeNgramFactory)
        self.assertEqual(features[0].base_name, "shape_ngram_3")
        self.assertEqual(res, "Xxx xX xxx")

    def test_word_cluster_factory(self):
        # Given
        resources = {
            WORD_CLUSTERS: {
                "my_word_clusters": {
                    "word1": "00",
                    "word2": "11"
                }
            }
        }

        config = {
            "factory_name": "word_cluster",
            "args": {
                "cluster_name": "my_word_clusters",
                "use_stemming": False
            },
            "offsets": [0]
        }

        tokens = tokenize("hello word1 word2", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        factory = CRFFeatureFactory.from_config(config, resources=resources)
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)
        features = factory.build_features()

        # When
        res0 = features[0].compute(0, cache)
        res1 = features[0].compute(1, cache)
        res2 = features[0].compute(2, cache)

        # Then
        self.assertIsInstance(factory, WordClusterFactory)
        self.assertEqual(features[0].base_name,
                         "word_cluster_my_word_clusters")
        self.assertEqual(res0, None)
        self.assertEqual(res1, "00")
        self.assertEqual(res2, "11")

    def test_entity_match_factory(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: my_intent
utterances:
- this is [entity1](my first entity)
- this is [entity2](second_entity)""")

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        config = {
            "factory_name": "entity_match",
            "args": {
                "tagging_scheme_code": TaggingScheme.BILOU.value,
                "use_stemming": True
            },
            "offsets": [0]
        }

        tokens = tokenize(
            "my first entity and second_entity and third_entity",
            LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        resources = {STEMS: dict()}
        custom_entity_parser = CustomEntityParser.build(
            dataset, CustomEntityParserUsage.WITH_STEMS, resources)
        factory = CRFFeatureFactory.from_config(
            config, custom_entity_parser=custom_entity_parser,
            resources=resources)
        factory.fit(dataset, "my_intent")

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
        self.assertIsInstance(factory, CustomEntityMatchFactory)
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].base_name, "entity_match_entity1")
        self.assertEqual(features[1].base_name, "entity_match_entity2")
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

    def test_entity_match_factory_with_filter(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: my_intent
utterances:
- this is [entity1](my first entity)
- this is [entity2](second_entity)
- this is [entity3](third_entity)

---
type: entity
name: entity3
automatically_extensible: false""")

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        config = {
            "factory_name": "entity_match",
            "args": {
                "tagging_scheme_code": TaggingScheme.BILOU.value,
                "use_stemming": True,
                "entity_filter": {
                    "automatically_extensible": True,
                    "invalid_filter": "i'm invalid"  # Should be ignored
                }
            },
            "offsets": [0]
        }

        tokens = tokenize(
            "my first entity and second_entity and third_entity", LANGUAGE_EN)
        cache = [{TOKEN_NAME: token} for token in tokens]
        resources = {STEMS: dict()}
        custom_entity_parser = CustomEntityParser.build(
            dataset, CustomEntityParserUsage.WITH_STEMS, resources)
        factory = CRFFeatureFactory.from_config(
            config, custom_entity_parser=custom_entity_parser,
            resources=resources)
        factory.fit(dataset, "my_intent")

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
        self.assertIsInstance(factory, CustomEntityMatchFactory)
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].base_name, "entity_match_entity1")
        self.assertEqual(features[1].base_name, "entity_match_entity2")
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
        def mock_builtin_entity_scope(dataset, _):
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
        builtin_entity_parser = BuiltinEntityParser.build(language="en")
        factory = CRFFeatureFactory.from_config(
            config, builtin_entity_parser=builtin_entity_parser)
        # pylint: disable=protected-access
        factory._get_builtin_entity_scope = mock_builtin_entity_scope
        # pylint: enable=protected-access
        mocked_dataset = {"language": "en"}
        factory.fit(mocked_dataset, None)

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

    def test_custom_single_feature_factory(self):
        # Given
        # pylint:disable=unused-variable
        @CRFFeatureFactory.register("my_single_feature", override=True)
        class MySingleFeatureFactory(SingleFeatureFactory):
            def compute_feature(self, tokens, token_index):
                return "(%s)[my_feature]" % tokens[token_index].value

        # pylint:enable=unused-variable

        # When
        config = {
            "factory_name": "my_single_feature",
            "args": {},
            "offsets": [0, -1]
        }
        feature_factory = CRFFeatureFactory.from_config(config)
        features = feature_factory.build_features()
        feature_name = features[0].name
        feature_name_offset = features[1].name
        tokens = tokenize("hello world", "en")
        cache = [{TOKEN_NAME: token} for token in tokens]
        feature_value = features[0].compute(1, cache)
        feature_value_offset = features[1].compute(1, cache)

        # Then
        self.assertEqual("my_single_feature", feature_name)
        self.assertEqual("my_single_feature[-1]", feature_name_offset)
        self.assertEqual("(world)[my_feature]", feature_value)
        self.assertEqual("(hello)[my_feature]", feature_value_offset)

    def test_custom_multi_feature_factory(self):
        # Given

        # pylint:disable=unused-variable
        @CRFFeatureFactory.register("my_multi_feature_factory", override=True)
        class MyMultiFeature(CRFFeatureFactory):
            def build_features(self):
                first_features = [
                    Feature("my_first_feature", self.compute_feature_1,
                            offset=offset) for offset in self.offsets]
                second_features = [
                    Feature("my_second_feature", self.compute_feature_2,
                            offset=offset) for offset in self.offsets]
                return first_features + second_features

            @staticmethod
            def compute_feature_1(tokens, token_index):
                return "(%s)[my_feature_1]" % tokens[token_index].value

            @staticmethod
            def compute_feature_2(tokens, token_index):
                return "(%s)[my_feature_2]" % tokens[token_index].value

        # pylint:enable=unused-variable

        # When
        config = {
            "factory_name": "my_multi_feature_factory",
            "args": {},
            "offsets": [-1, 0]
        }
        feature_factory = CRFFeatureFactory.from_config(config)
        features = feature_factory.build_features()
        feature_0 = features[0]
        feature_1 = features[1]
        feature_2 = features[2]
        feature_3 = features[3]
        tokens = tokenize("foo bar baz", "en")
        cache = [{TOKEN_NAME: token} for token in tokens]

        # Then
        self.assertEqual("my_first_feature[-1]", feature_0.name)
        self.assertEqual("(foo)[my_feature_1]", feature_0.compute(1, cache))
        self.assertEqual("my_first_feature", feature_1.name)
        self.assertEqual("my_second_feature[-1]", feature_2.name)
        self.assertEqual("(bar)[my_feature_2]", feature_2.compute(2, cache))
        self.assertEqual("my_second_feature", feature_3.name)

    def test_factory_from_config(self):
        # Given
        @CRFFeatureFactory.register("my_custom_feature")
        class MySingleFeatureFactory(SingleFeatureFactory):
            def compute_feature(self, tokens, token_index):
                return "(%s)[my_custom_feature]" % tokens[token_index].value

        config = {
            "factory_name": "my_custom_feature",
            "args": {},
            "offsets": [0]
        }

        # When
        factory = CRFFeatureFactory.from_config(config)

        # Then
        self.assertIsInstance(factory, MySingleFeatureFactory)

    def test_should_fail_loading_unregistered_factory_from_config(self):
        config = {
            "factory_name": "my_unknown_feature",
            "args": {},
            "offsets": [0]
        }

        # When / Then
        with self.assertRaises(NotRegisteredError):
            CRFFeatureFactory.from_config(config)
