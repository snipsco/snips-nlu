# coding=utf-8
from __future__ import unicode_literals

import io

import numpy as np
from builtins import str, zip
from mock import patch
from snips_nlu_utils import normalize

from snips_nlu.constants import LANGUAGE_EN
from snips_nlu.dataset import validate_and_format_dataset, Dataset
from snips_nlu.entity_parser import BuiltinEntityParser, CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser_usage import (
    CustomEntityParserUsage)
from snips_nlu.exceptions import DatasetFormatError
from snips_nlu.intent_classifier.featurizer import (
    CooccurrenceVectorizer, Featurizer, TfidfVectorizer)
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    text_to_utterance)
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.pipeline.configs.intent_classifier import (
    CooccurrenceVectorizerConfig)
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.tests.utils import FixtureTest
from snips_nlu.common.utils import json_string


def _stem(t):
    t = normalize(t)
    if t == "beautiful":
        s = "beauty"
    elif t == "birdy":
        s = "bird"
    elif t == "entity":
        s = "ent"
    else:
        s = t
    return s


def stem_function(text, language):
    return get_default_sep(language).join(
        [_stem(t) for t in tokenize_light(text, language)])


# pylint: disable=protected-access
class TestIntentClassifierFeaturizer(FixtureTest):
    def test_should_be_serializable(self):
        # Given
        pvalue_threshold = 0.42
        config = FeaturizerConfig(pvalue_threshold=pvalue_threshold,
                                  word_clusters_name="brown_clusters",
                                  added_cooccurrence_feature_ratio=0.2)
        featurizer = Featurizer(
            config=config,
            unknown_words_replacement_string=None)

        dataset_stream = io.StringIO("""
---
type: intent
name: dummy_intent
utterances:
  - this is the number [number:snips/number](one)
""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        dataset = validate_and_format_dataset(dataset)
        utterances = [text_to_utterance("this is the number"),
                      text_to_utterance("yo")]
        classes = np.array([0, 1])
        featurizer.fit(dataset, utterances, classes, max(classes))

        # When
        featurizer.persist(self.tmp_file_path)

        # Then
        expected_featurizer_dict = {
            "language_code": "en",
            "tfidf_vectorizer": "tfidf_vectorizer",
            "cooccurrence_vectorizer": "cooccurrence_vectorizer",
            "config": config.to_dict(),
            "builtin_entity_scope": ["snips/number"]
        }
        featurizer_dict_path = self.tmp_file_path / "featurizer.json"
        self.assertJsonContent(featurizer_dict_path, expected_featurizer_dict)

        expected_metadata = {"unit_name": "featurizer"}
        metadata_path = self.tmp_file_path / "metadata.json"
        self.assertJsonContent(metadata_path, expected_metadata)

        tfidf_vectorizer_path = self.tmp_file_path / "tfidf_vectorizer"
        self.assertTrue(tfidf_vectorizer_path.exists())

        cooccurrence_vectorizer_path = (
                self.tmp_file_path / "cooccurrence_vectorizer")
        self.assertTrue(cooccurrence_vectorizer_path.exists())

    def test_should_be_serializable_before_fit(self):
        # Given
        pvalue_threshold = 0.42
        config = FeaturizerConfig(pvalue_threshold=pvalue_threshold,
                                  word_clusters_name="brown_clusters",
                                  added_cooccurrence_feature_ratio=0.2)
        featurizer = Featurizer(
            config=config,
            unknown_words_replacement_string=None)

        # When
        featurizer.persist(self.tmp_file_path)

        # Then
        expected_featurizer_dict = {
            "language_code": None,
            "tfidf_vectorizer": None,
            "cooccurrence_vectorizer": None,
            "config": config.to_dict(),
            "builtin_entity_scope": None
        }
        featurizer_dict_path = self.tmp_file_path / "featurizer.json"
        self.assertJsonContent(featurizer_dict_path, expected_featurizer_dict)

        expected_metadata = {"unit_name": "featurizer"}
        metadata_path = self.tmp_file_path / "metadata.json"
        self.assertJsonContent(metadata_path, expected_metadata)

        tfidf_vectorizer_path = self.tmp_file_path / "tfidf_vectorizer"
        self.assertFalse(tfidf_vectorizer_path.exists())

        cooccurrence_vectorizer_path = (
                self.tmp_file_path / "cooccurrence_vectorizer")
        self.assertFalse(cooccurrence_vectorizer_path.exists())

    @patch("snips_nlu.intent_classifier.featurizer.TfidfVectorizer.from_path")
    @patch("snips_nlu.intent_classifier.featurizer.CooccurrenceVectorizer"
           ".from_path")
    def test_should_be_deserializable(self, mocked_cooccurrence_load,
                                      mocked_tfidf_load):
        # Given
        mocked_tfidf_load.return_value = "tfidf_vectorizer"
        mocked_cooccurrence_load.return_value = "cooccurrence_vectorizer"

        language = LANGUAGE_EN
        config = FeaturizerConfig()

        builtin_scope = ["snips/datetime"]
        none_class_ix = 2
        featurizer_dict = {
            "language_code": language,
            "tfidf_vectorizer": "tfidf_vectorizer",
            "cooccurrence_vectorizer": "cooccurrence_vectorizer",
            "config": config.to_dict(),
            "builtin_entity_scope": builtin_scope
        }

        self.tmp_file_path.mkdir()
        featurizer_path = self.tmp_file_path / "featurizer.json"
        with featurizer_path.open("w", encoding="utf-8") as f:
            f.write(json_string(featurizer_dict))

        # When
        featurizer = Featurizer.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(language, featurizer.language)
        self.assertEqual(set(builtin_scope), featurizer.builtin_entity_scope)
        self.assertEqual("tfidf_vectorizer", featurizer.tfidf_vectorizer)
        self.assertEqual("cooccurrence_vectorizer",
                         featurizer.cooccurrence_vectorizer)
        self.assertDictEqual(config.to_dict(), featurizer.config.to_dict())

    @patch("snips_nlu.intent_classifier.featurizer.get_word_cluster")
    @patch("snips_nlu.intent_classifier.featurizer.stem")
    @patch("snips_nlu.entity_parser.custom_entity_parser.stem")
    def test_preprocess(self, mocked_parser_stem, mocked_featurizer_stem,
                        mocked_word_cluster):
        # Given
        language = LANGUAGE_EN

        mocked_word_cluster.return_value = {
            "beautiful": "cluster_1",
            "birdy": "cluster_2",
            "entity": "cluster_3"
        }

        mocked_parser_stem.side_effect = stem_function
        mocked_featurizer_stem.side_effect = stem_function

        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
    - dummy utterance

---
type: entity
name: entity_1
automatically_extensible: false
use_synononyms: false
matching_strictness: 1.0
values:
  - [entity 1, alternative entity 1]
  - [éntity 1, alternative entity 1]
  
---
type: entity
name: entity_2
automatically_extensible: false
use_synononyms: true
matching_strictness: 1.0
values:
  - entity 1
  - [Éntity 2, Éntity_2, Alternative entity 2]
""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        dataset = validate_and_format_dataset(dataset)

        custom_entity_parser = CustomEntityParser.build(
            dataset, CustomEntityParserUsage.WITH_STEMS)

        builtin_entity_parser = BuiltinEntityParser.build(dataset, language)
        utterances = [
            text_to_utterance("hÉllo wOrld Éntity_2"),
            text_to_utterance("beauTiful World entity 1"),
            text_to_utterance("Bird bïrdy"),
            text_to_utterance("Bird birdy"),
        ]

        config = FeaturizerConfig(
            use_stemming=True, word_clusters_name="brown_clusters")
        featurizer = Featurizer(language=language,
                                custom_entity_parser=custom_entity_parser,
                                builtin_entity_parser=builtin_entity_parser,
                                config=config)
        featurizer.language = language

        # When
        processed_data = featurizer._preprocess(
            utterances, training=False)
        processed_data = list(zip(*processed_data))

        # Then
        u_0 = {
            "data": [
                {
                    "text": "hello world entity_2"
                }
            ]
        }

        u_1 = {
            "data": [
                {
                    "text": "beauty world ent 1"
                }
            ]
        }

        u_2 = {
            "data": [
                {
                    "text": "bird bird"
                }
            ]
        }

        u_3 = {
            "data": [
                {
                    "text": "bird bird"
                }
            ]
        }

        ent_0 = {
            "entity_kind": "entity_2",
            "value": "entity_2",
            "resolved_value": "Éntity 2",
            "range": {"start": 12, "end": 20}
        }
        num_0 = {
            "entity_kind": "snips/number",
            "value": "2",
            "resolved_value": {
                "value": 2.0,
                "kind": "Number"
            },
            "range": {"start": 19, "end": 20}
        }
        ent_11 = {
            "entity_kind": "entity_1",
            "value": "ent 1",
            "resolved_value": "entity 1",
            "range": {"start": 13, "end": 18}
        }
        ent_12 = {
            "entity_kind": "entity_2",
            "value": "ent 1",
            "resolved_value": "entity 1",
            "range": {"start": 13, "end": 18}
        }
        num_1 = {
            "entity_kind": "snips/number",
            "value": "1",
            "range": {"start": 17, "end": 18},
            "resolved_value": {
                "value": 1.0,
                "kind": "Number"
            },
        }

        expected_data = [
            (u_0,
             "hello world entity_2",
             [num_0],
             [ent_0],
             []
             ),
            (
                u_1,
                "beauty world ent 1",
                [num_1],
                [ent_11, ent_12],
                ["cluster_1", "cluster_3"]
            ),
            (
                u_2,
                "bird bird",
                [],
                [],
                []
            ),
            (
                u_3,
                "bird bird",
                [],
                [],
                ["cluster_2"]
            )
        ]

        self.assertSequenceEqual(expected_data, processed_data)

    @patch("snips_nlu.intent_classifier.featurizer.get_word_cluster")
    @patch("snips_nlu.intent_classifier.featurizer.stem")
    def test_preprocess_for_training(self, mocked_featurizer_stem,
                                     mocked_word_cluster):
        # Given
        language = LANGUAGE_EN

        mocked_word_cluster.return_value = {
            "beautiful": "cluster_1",
            "birdy": "cluster_2",
            "entity": "cluster_3"
        }

        mocked_featurizer_stem.side_effect = stem_function

        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
    - dummy utterance

---
type: entity
name: entity_1
automatically_extensible: false
use_synononyms: false
matching_strictness: 1.0
values:
  - [entity 1, alternative entity 1]
  - [éntity 1, alternative entity 1]
  
---
type: entity
name: entity_2
automatically_extensible: false
use_synononyms: true
matching_strictness: 1.0
values:
  - entity 1
  - [Éntity 2, Éntity_2, Alternative entity 2]
""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        dataset = validate_and_format_dataset(dataset)

        custom_entity_parser = CustomEntityParser.build(
            dataset, CustomEntityParserUsage.WITH_STEMS)

        builtin_entity_parser = BuiltinEntityParser.build(dataset, language)
        utterances = [
            {
                "data": [
                    {
                        "text": "hÉllo wOrld "
                    },
                    {
                        "text": " yo "
                    },
                    {
                        "text": " yo "
                    },
                    {
                        "text": "yo "
                    },
                    {
                        "text": "Éntity_2 ",
                        "entity": "entity_2"
                    },
                    {
                        "text": "Éntity_2",
                        "entity": "entity_2"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "beauTiful World "
                    },
                    {
                        "text": "entity 1",
                        "entity": "entity_1"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "Bird bïrdy"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "Bird birdy"
                    }
                ]
            }
        ]

        config = FeaturizerConfig(
            use_stemming=True, word_clusters_name="brown_clusters")
        featurizer = Featurizer(language=language,
                                custom_entity_parser=custom_entity_parser,
                                builtin_entity_parser=builtin_entity_parser,
                                config=config)
        featurizer.language = language

        # When
        processed_data = featurizer._preprocess(
            utterances, training=True)
        processed_data = list(zip(*processed_data))

        # Then
        u_0 = {
            "data": [
                {
                    "text": "hello world"
                },
                {
                    "text": "yo"
                },
                {
                    "text": "yo"
                },
                {
                    "text": "yo"
                },
                {
                    "text": "entity_2",
                    "entity": "entity_2"
                },
                {
                    "text": "entity_2",
                    "entity": "entity_2"
                }
            ]
        }

        u_1 = {
            "data": [
                {
                    "text": "beauty world"
                },
                {
                    "text": "ent 1",
                    "entity": "entity_1"
                }
            ]
        }
        u_2 = {
            "data": [
                {
                    "text": "bird bird"
                }
            ]
        }

        ent_00 = {
            "entity_kind": "entity_2",
            "value": "entity_2",
            "range": {"start": 21, "end": 29}
        }
        ent_01 = {
            "entity_kind": "entity_2",
            "value": "entity_2",
            "range": {"start": 30, "end": 38}
        }
        ent_1 = {
            "entity_kind": "entity_1",
            "value": "ent 1",
            "range": {"start": 13, "end": 18}
        }

        expected_data = [
            (
                u_0,
                "hello world yo yo yo entity_2 entity_2",
                [],
                [ent_00, ent_01],
                []
            ),
            (
                u_1,
                "beauty world ent 1",
                [],
                [ent_1],
                ["cluster_1", "cluster_3"]
            ),
            (
                u_2,
                "bird bird",
                [],
                [],
                []
            ),
            (
                u_2,
                "bird bird",
                [],
                [],
                ["cluster_2"]
            )
        ]

        self.assertSequenceEqual(expected_data, processed_data)

    def test_featurizer_should_be_serialized_when_not_fitted(self):
        # Given
        featurizer = Featurizer()
        # When/Then
        featurizer.persist(self.tmp_file_path)

    def test_fit_transform_should_be_consistent_with_transform(self):
        # Here we mainly test that the output of fit_transform is
        # the same as the result of fit and then transform.
        # We're trying to avoid that for some reason indexes of features
        # get mixed up after feature selection

        # Given
        config = FeaturizerConfig(added_cooccurrence_feature_ratio=.5)
        featurizer = Featurizer(config=config)

        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
    - dummy utterance

---
type: entity
name: entity_1
automatically_extensible: false
use_synononyms: false
matching_strictness: 1.0
values:
  - [entity 1, alternative entity 1]
  - [éntity 1, alternative entity 1]

---
type: entity
name: entity_2
automatically_extensible: false
use_synononyms: true
matching_strictness: 1.0
values:
  - entity 1
  - [Éntity 2, Éntity_2, Alternative entity 2]
        """)
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        dataset = validate_and_format_dataset(dataset)

        utterances = [
            {
                "data": [
                    {
                        "text": "hÉllo wOrld "
                    },
                    {
                        "text": "Éntity_2",
                        "entity": "entity_2"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "beauTiful World "
                    },
                    {
                        "text": "entity 1",
                        "entity": "entity_1"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "Bird bïrdy"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "Bird bïrdy"
                    }
                ]
            }
        ]

        classes = [0, 0, 1, 1]

        # When
        x_0 = featurizer.fit_transform(dataset, utterances, classes,
                                       max(classes))
        x_1 = featurizer.transform(utterances)

        # Then
        self.assertListEqual(x_0.todense().tolist(), x_1.todense().tolist())

    def test_fit_with_no_utterance_should_raise(self):
        # Given
        utterances = []
        classes = []
        dataset = {"language": "en"}

        # When/Then
        with self.assertRaises(DatasetFormatError) as ctx:
            Featurizer().fit_transform(dataset, utterances, classes, None)

        self.assertEqual("Tokenized utterances are empty", str(ctx.exception))

    def test_feature_index_to_feature_name(self):
        # Given
        config = FeaturizerConfig(added_cooccurrence_feature_ratio=.75)
        featurizer = Featurizer(config=config)

        self.assertDictEqual(dict(), featurizer.feature_index_to_feature_name)

        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
    - dummy utterance

---
type: entity
name: entity_1
automatically_extensible: false
use_synononyms: false
matching_strictness: 1.0
values:
  - [entity 1, alternative entity 1]
  - [éntity 1, alternative entity 1]

---
type: entity
name: entity_2
automatically_extensible: false
use_synononyms: true
matching_strictness: 1.0
values:
  - entity 1
  - [Éntity 2, Éntity_2, Alternative entity 2]
        """)
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        dataset = validate_and_format_dataset(dataset)

        utterances = [
            {
                "data": [
                    {
                        "text": "hÉllo wOrld "
                    },
                    {
                        "text": "Éntity_2",
                        "entity": "entity_2"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "beauTiful World "
                    },
                    {
                        "text": "entity 1",
                        "entity": "entity_1"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "Bird bïrdy"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "Bird bïrdy"
                    }
                ]
            }
        ]

        classes = [0, 0, 1, 1]

        # When
        featurizer.fit(dataset, utterances, classes, max(classes))

        # Then
        expected = {
            0: "ngram:bird",
            1: "ngram:birdy",
            2: "ngram:world",
            3: u'pair:world+ENTITY_1',
            4: "pair:world+ENTITY_2"
        }
        self.assertDictEqual(
            expected, featurizer.feature_index_to_feature_name)


class CooccurrenceVectorizerTest(FixtureTest):

    def test_cooccurrence_vectorizer_should_persist(self):
        # Given
        x = [("yo yo", [], [])]
        language = "en"
        vectorizer = CooccurrenceVectorizer().fit(x, language)

        # When
        vectorizer.persist(self.tmp_file_path)

        # Then
        metadata_path = self.tmp_file_path / "metadata.json"
        expected_metadata = {"unit_name": "cooccurrence_vectorizer"}
        self.assertJsonContent(metadata_path, expected_metadata)

        vectorizer_path = self.tmp_file_path / "vectorizer.json"
        expected_vectorizer = {
            "word_pairs": {
                "0": ["yo", "yo"]
            },
            "language_code": "en",
            "config": vectorizer.config.to_dict()
        }
        self.assertJsonContent(vectorizer_path, expected_vectorizer)

    def test_cooccurrence_vectorizer_should_load(self):
        # Given
        config = CooccurrenceVectorizerConfig()

        word_pairs = {
            ("a", "b"): 0,
            ("a", 'c'): 12
        }

        serializable_word_pairs = {
            0: ["a", "b"],
            12: ["a", "c"]
        }

        vectorizer_dict = {
            "unit_name": "cooccurrence_vectorizer",
            "language_code": "en",
            "word_pairs": serializable_word_pairs,
            "config": config.to_dict(),
        }

        self.tmp_file_path.mkdir()
        self.writeJsonContent(
            self.tmp_file_path / "vectorizer.json",
            vectorizer_dict
        )

        # When
        vectorizer = CooccurrenceVectorizer.from_path(self.tmp_file_path)

        # Then
        self.assertDictEqual(config.to_dict(), vectorizer.config.to_dict())
        self.assertEqual("en", vectorizer.language)
        self.assertDictEqual(vectorizer.word_pairs, word_pairs)

    def test_preprocess(self):
        # Given
        u = "a b c d e f"
        builtin_ents = [
            {
                "value": "e",
                "resolved_value": "e",
                "range": {
                    "start": 8,
                    "end": 9
                },
                "entity_kind": "the_snips_e_entity"
            }
        ]
        custom_ents = [
            {
                "value": "c",
                "resolved_value": "c",
                "range": {
                    "start": 4,
                    "end": 5
                },
                "entity_kind": "the_c_entity"
            }
        ]

        x = [(u, builtin_ents, custom_ents)]
        vectorizer = CooccurrenceVectorizer()
        vectorizer._language = "en"

        # When
        preprocessed = vectorizer._preprocess(x)

        # Then
        expected = [["a", "b", "THE_C_ENTITY", "d", "THE_SNIPS_E_ENTITY", "f"]]
        self.assertSequenceEqual(expected, preprocessed)

    def test_transform(self):
        # Given
        config = CooccurrenceVectorizerConfig(
            filter_stop_words=True,
            window_size=3,
            unknown_words_replacement_string="d")
        vectorizer = CooccurrenceVectorizer(config)
        vectorizer._language = "en"
        vectorizer._word_pairs = {
            ("THE_SNIPS_E_ENTITY", "f"): 0,
            ("a", "THE_C_ENTITY"): 1,
            ("a", "THE_SNIPS_E_ENTITY"): 2,
            ("b", "THE_SNIPS_E_ENTITY"): 3,
            ("yo", "yo"): 4,
            ("d", "THE_SNIPS_E_ENTITY"): 5
        }

        u = "yo a b c d e f yo"
        builtin_ents = [
            {
                "value": "e",
                "resolved_value": "e",
                "range": {
                    "start": 11,
                    "end": 12
                },
                "entity_kind": "the_snips_e_entity"
            }
        ]
        custom_ents = [
            {
                "value": "c",
                "resolved_value": "c",
                "range": {
                    "start": 7,
                    "end": 8
                },
                "entity_kind": "the_c_entity"
            }
        ]

        data = [
            (u, builtin_ents, custom_ents),
            (u[:-5], builtin_ents, custom_ents),
        ]

        # When
        with patch("snips_nlu.intent_classifier.featurizer.get_stop_words") \
                as mocked_stop_words:
            mocked_stop_words.return_value = {"b"}
            x = vectorizer.transform(data)
        # Then
        expected = [[1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]]
        self.assertEqual(expected, x.todense().tolist())

    def test_fit(self):
        u = "a b c d e f"
        builtin_ents = [
            {
                "value": "e",
                "resolved_value": "e",
                "range": {
                    "start": 8,
                    "end": 9
                },
                "entity_kind": "the_snips_e_entity"
            }
        ]
        custom_ents = [
            {
                "value": "c",
                "resolved_value": "c",
                "range": {
                    "start": 4,
                    "end": 5
                },
                "entity_kind": "the_c_entity"
            }
        ]

        x = [(u, builtin_ents, custom_ents)]

        config = CooccurrenceVectorizerConfig(
            window_size=3,
            unknown_words_replacement_string="b"
        )
        language = "en"

        # When
        expected_pairs = {
            ("THE_C_ENTITY", "THE_SNIPS_E_ENTITY"): 0,
            ("THE_C_ENTITY", "d"): 1,
            ("THE_C_ENTITY", "f"): 2,
            ("THE_SNIPS_E_ENTITY", "f"): 3,
            ("a", "THE_C_ENTITY"): 4,
            ("a", "THE_SNIPS_E_ENTITY"): 5,
            ("a", "d"): 6,
            ("d", "THE_SNIPS_E_ENTITY"): 7,
            ("d", "f"): 8,
        }
        vectorizer = CooccurrenceVectorizer(config).fit(x, language)

        # Then
        self.assertDictEqual(expected_pairs, vectorizer.word_pairs)

    def test_limit_vocabulary(self):
        # Given
        config = CooccurrenceVectorizerConfig(filter_stop_words=False)
        vectorizer = CooccurrenceVectorizer(config=config)
        train_data = [
            ("a b", [], []),
            ("a c", [], []),
            ("a d", [], []),
            ("a e", [], []),
        ]
        data = [
            ("a c e", [], []),
            ("a d e", [], []),
        ]

        vectorizer.fit(train_data, "en")
        x_0 = vectorizer.transform(data)
        pairs = {
            ("a", "b"): 0,
            ("a", "c"): 1,
            ("a", "d"): 2,
            ("a", "e"): 3
        }
        kept_pairs = [("a", "b"), ("a", "c"), ("a", "d")]
        self.assertDictEqual(pairs, vectorizer.word_pairs)

        # When

        kept_pairs_indexes = [pairs[p] for p in kept_pairs]
        vectorizer.limit_word_pairs(kept_pairs)

        # Then
        expected_pairs = {
            ("a", "b"): 0,
            ("a", "c"): 1,
            ("a", "d"): 2
        }
        self.assertDictEqual(expected_pairs, vectorizer.word_pairs)
        x_1 = vectorizer.transform(data)
        self.assertListEqual(
            x_0[:, kept_pairs_indexes].todense().tolist(),
            x_1.todense().tolist()
        )

    def test_limit_vocabulary_should_raise(self):
        # Given
        vectorizer = CooccurrenceVectorizer()
        vectorizer._word_pairs = {
            ("a", "b"): 0,
            ("a", "c"): 1,
            ("a", "d"): 2,
            ("a", "e"): 3
        }

        # When / Then
        with self.assertRaises(ValueError) as ctx:
            vectorizer.limit_word_pairs([("a", "f")])

        self.assertEqual(
            str(ctx.exception),
            "Invalid word pairs [(u'a', u'f')], expected values in"
            " [(u'a', u'b'), (u'a', u'c'), (u'a', u'd'), (u'a', u'e')]"
        )


class TestTfidfVectorizer(FixtureTest):

    def test_enrich_utterance_for_tfidf(self):
        # Given
        utterances = [
            {
                "data": [
                    {
                        "text": "one",
                        "entity": "snips/number"
                    },
                    {
                        "text": "beauty world",
                    },
                    {
                        "text": "ent 1",
                        "entity": "dummy_entity_1"
                    },
                ]
            },
            text_to_utterance("one beauty world ent 1"),
            text_to_utterance("hello world entity_2"),
            text_to_utterance("bird bird"),
        ]

        builtin_ents = [
            [
                {
                    "value": "one",
                    "resolved_value": 1,
                    "range": {
                        "start": 0,
                        "end": 3
                    },
                    "entity_kind": "snips/number"

                }
            ],
            [
                {
                    "value": "one",
                    "resolved_value": 1,
                    "range": {
                        "start": 0,
                        "end": 3
                    },
                    "entity_kind": "snips/number"
                },
                {
                    "value": "1",
                    "resolved_value": 1,
                    "range": {
                        "start": 27,
                        "end": 28
                    },
                    "entity_kind": "snips/number"
                }
            ],
            [
                {
                    "value": "2",
                    "resolved_value": 2,
                    "range": {
                        "start": 19,
                        "end": 20
                    },
                    "entity_kind": "snips/number"
                }
            ],
            []
        ]

        custom_ents = [
            [
                {
                    "value": "ent 1",
                    "resolved_value": "entity 1",
                    "range": {
                        "start": 20,
                        "end": 28
                    },
                    "entity_kind": "dummy_entity_1"
                }
            ],
            [

                {
                    "value": "ent 1",
                    "resolved_value": "entity 1",
                    "range": {
                        "start": 20,
                        "end": 28
                    },
                    "entity_kind": "dummy_entity_1"
                }
            ],
            [
                {
                    "value": "entity_2",
                    "resolved_value": "Éntity_2",
                    "range": {
                        "start": 12,
                        "end": 20
                    },
                    "entity_kind": "dummy_entity_2"
                }
            ],
            []
        ]

        w_clusters = [
            ["111", "112"],
            ["111", "112"],
            [],
            []
        ]

        vectorizer = TfidfVectorizer()
        vectorizer._language = "en"

        # When
        enriched_utterances = [
            vectorizer._enrich_utterance_for_tfidf(*data)
            for data in zip(utterances, builtin_ents, custom_ents, w_clusters)
        ]

        # Then
        expected_u0 = "beauty world ent 1 " \
                      "builtinentityfeaturesnipsnumber " \
                      "entityfeaturedummy_entity_1 111 112"

        expected_u1 = "one beauty world ent 1 " \
                      "builtinentityfeaturesnipsnumber " \
                      "builtinentityfeaturesnipsnumber " \
                      "entityfeaturedummy_entity_1 111 112"

        expected_u2 = "hello world entity_2 builtinentityfeaturesnipsnumber " \
                      "entityfeaturedummy_entity_2"

        expected_u3 = "bird bird"

        expected_utterances = [
            expected_u0,
            expected_u1,
            expected_u2,
            expected_u3
        ]

        self.assertEqual(expected_utterances, enriched_utterances)

    def test_limit_vocabulary(self):
        # Given
        vectorizer = TfidfVectorizer()
        language = "en"

        utterances = [
            (text_to_utterance("5 55 6 66 666"), [], [], []),
            (text_to_utterance("55 66"), [], [], [])
        ]

        voca = {
            "5": 0,
            "55": 1,
            "6": 2,
            "66": 3,
            "666": 4
        }
        kept_unigrams = ["5", "6", "666"]
        vectorizer.fit(utterances, language)
        self.assertDictEqual(voca, vectorizer.vocabulary)
        diag = vectorizer.idf_diag.copy()

        # When
        vectorizer.limit_vocabulary(kept_unigrams)

        # Then
        expected_voca = {
            "5": 0,
            "6": 1,
            "666": 2
        }
        self.assertDictEqual(expected_voca, vectorizer.vocabulary)

        expected_diag = diag[[voca[u] for u in kept_unigrams]].tolist()
        self.assertListEqual(expected_diag, vectorizer.idf_diag.tolist())

    def test_limit_vocabulary_should_raise(self):
        # Given
        vectorizer = TfidfVectorizer()
        language = "en"
        utterances = [(text_to_utterance("5 55 6 66 666"), [], [], [])]

        vectorizer.fit(utterances, language)

        # When / Then
        kept_indexes = ["7", "8"]
        with self.assertRaises(ValueError) as ctx:
            vectorizer.limit_vocabulary(kept_indexes)

        expected_ctx = "Invalid ngrams [u'7', u'8'], expected values" \
                       " in [u'5', u'55', u'6', u'66', u'666']"
        self.assertEqual(expected_ctx, str(ctx.exception))
