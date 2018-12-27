# coding=utf-8
from __future__ import unicode_literals

import json
from builtins import str, zip

import numpy as np
from mock import patch
from scipy.sparse import spdiags
from snips_nlu_utils import normalize

from snips_nlu.constants import LANGUAGE_EN
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.entity_parser import BuiltinEntityParser, CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser_usage import (
    CustomEntityParserUsage)
from snips_nlu.exceptions import _EmptyDataError
from snips_nlu.intent_classifier.featurizer import (
    CooccurrenceVectorizer, Featurizer, _get_tfidf_vectorizer)
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    text_to_utterance)
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.pipeline.configs.intent_classifier import (
    CooccurrenceVectorizerConfig)
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.tests.utils import FixtureTest, SAMPLE_DATASET
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
        featurizer = Featurizer(
            config=FeaturizerConfig(pvalue_threshold=pvalue_threshold,
                                    word_clusters_name="brown_clusters"),
            unknown_words_replacement_string=None)
        dataset = {
            "entities": {
                "entity2": {
                    "data": [
                        {
                            "value": "entity1",
                            "synonyms": ["entity1"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": True,
                    "matching_strictness": 1.0
                }
            },
            "intents": {},
            "language": "en"
        }
        dataset = validate_and_format_dataset(dataset)

        utterances = [
            "hello world",
            "beautiful world",
            "hello here",
            "bird birdy",
            "beautiful bird"
        ]
        utterances = [text_to_utterance(u) for u in utterances]
        classes = np.array([0, 0, 0, 1, 1])

        featurizer.fit(dataset, utterances, classes)

        # When
        serialized_featurizer = featurizer.to_dict()

        # Then
        msg = "Featurizer dict should be json serializable to utf8."
        with self.fail_if_exception(msg):
            dumped = json_string(serialized_featurizer)

        msg = "SnipsNLUEngine should be deserializable from dict with " \
              "unicode values"
        with self.fail_if_exception(msg):
            _ = Featurizer.from_dict(json.loads(dumped))

        vocabulary = featurizer.tfidf_vectorizer.vocabulary_
        # pylint: disable=W0212
        idf_diag = featurizer.tfidf_vectorizer._tfidf._idf_diag.data.tolist()
        # pylint: enable=W0212

        expected_serialized = {
            "config": {
                "sublinear_tf": False,
                "pvalue_threshold": pvalue_threshold,
                "added_cooccurrence_feature_ratio": 0,
                "word_clusters_name": "brown_clusters",
                "use_stemming": False
            },
            "language_code": "en",
            "tfidf_vectorizer": {"idf_diag": idf_diag, "vocab": vocabulary},
            "unknown_words_replacement_string": None,
            "builtin_entity_scope": []
        }
        self.assertDictEqual(expected_serialized, serialized_featurizer)

    def test_should_be_deserializable(self):
        # Given
        language = LANGUAGE_EN
        idf_diag = [1.52, 1.21, 1.04]
        vocabulary = {"hello": 0, "beautiful": 1, "world": 2}

        config = {
            "pvalue_threshold": 0.4,
            "sublinear_tf": False,
            "added_cooccurrence_feature_ratio": 0,
            "word_clusters_name": "brown_clusters",
            "use_stemming": False
        }

        featurizer_dict = {
            "config": config,
            "language_code": language,
            "tfidf_vectorizer": {"idf_diag": idf_diag, "vocab": vocabulary},
            "unknown_words_replacement_string": None,
            "builtin_entity_scope": None
        }

        # When
        featurizer = Featurizer.from_dict(featurizer_dict)

        # Then
        self.assertEqual(featurizer.language, language)
        # pylint: disable=W0212
        self.assertListEqual(
            featurizer.tfidf_vectorizer._tfidf._idf_diag.data.tolist(),
            idf_diag)
        # pylint: enable=W0212
        self.assertDictEqual(featurizer.tfidf_vectorizer.vocabulary_,
                             vocabulary)
        self.assertEqual(config, featurizer.config.to_dict())

    @patch("snips_nlu.intent_classifier.featurizer.stem")
    def test_enrich_utterance_for_tfidf(self, mocked_stem):
        # Given
        mocked_stem.side_effect = stem_function
        utterances = [
            {
                "data": [
                    {
                        "text": "one",
                        "entity": "snips/number"
                    },
                    {
                        "text": " beauTiful World ",
                    },
                    {
                        "text": "entity 1",
                        "entity": "dummy_entity_1"
                    },
                ]
            },
            text_to_utterance("one beauTiful World entity 1"),
            text_to_utterance("hÉllo wOrld Éntity_2"),
            text_to_utterance("Bird bïrdy"),
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

        featurizer = Featurizer(
            config=FeaturizerConfig(word_clusters_name="brown_clusters",
                                    use_stemming=True))
        featurizer.language = "en"

        # When
        enriched_utterances = [
            featurizer._enrich_utterance_for_tfidf(*data)
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

        dataset = {
            "intents": {
                "intent1": {
                    "utterances": []
                }
            },
            "entities": {
                "entity_1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["alternative entity 1"]
                        },
                        {
                            "value": "éntity 1",
                            "synonyms": ["alternative entity 1"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": False,
                    "matching_strictness": 1.0
                },
                "entity_2": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": []
                        },
                        {
                            "value": "Éntity 2",
                            "synonyms": ["Éntity_2", "Alternative entity 2"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "matching_strictness": 1.0
                },
                "snips/number": {}  # To test the builtin feature
            },
            "language": "en",
        }

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
        ent_0 = {
            "entity_kind": "entity_2",
            "value": "entity_2",
            "resolved_value": "Éntity 2",
            "range": {"start": 12, "end": 20}
        }
        num_0 = {
            "entity_kind": "snips/number",
            "value": "2",
            "entity": {
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
            "entity": {
                "value": 1.0,
                "kind": "Number"
            },
        }

        expected_data = [
            (
                "hello world entity_2",
                [num_0],
                [ent_0],
                []
            ),
            (
                "beauty world ent 1",
                [num_1],
                [ent_11, ent_12],
                ["cluster_1", "cluster_3"]
            ),
            (
                "bird bird",
                [],
                [],
                []
            ),
            (
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

        dataset = {
            "intents": {
                "intent1": {
                    "utterances": []
                }
            },
            "entities": {
                "entity_1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["alternative entity 1"]
                        },
                        {
                            "value": "éntity 1",
                            "synonyms": ["alternative entity 1"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": False,
                    "matching_strictness": 1.0
                },
                "entity_2": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": []
                        },
                        {
                            "value": "Éntity 2",
                            "synonyms": ["Éntity_2", "Alternative entity 2"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "matching_strictness": 1.0
                },
                "snips/number": {}  # To test the builtin feature
            },
            "language": "en",
        }

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
                "hello world yo yo yo entity_2 entity_2",
                [],
                [ent_00, ent_01],
                []
            ),
            (
                "beauty world ent 1",
                [],
                [ent_1],
                ["cluster_1", "cluster_3"]
            ),
            (
                "bird bird",
                [],
                [],
                []
            ),
            (
                "bird bird",
                [],
                [],
                ["cluster_2"]
            )
        ]

        self.assertSequenceEqual(expected_data, processed_data)

    def test_featurizer_should_be_serialized_when_not_fitted(self):
        # Given
        language = LANGUAGE_EN
        featurizer = Featurizer(language)
        # When/Then
        featurizer.to_dict()

    def test_limit_tfidf_vectorizer_vocabulary(self):
        # Given
        featurizer = Featurizer()
        featurizer.tfidf_vectorizer = _get_tfidf_vectorizer("en")

        utterances = [
            "5 55 6 66 666",
            "55 66",
        ]

        voca = {
            "5": 0,
            "55": 1,
            "6": 2,
            "66": 3,
            "666": 4
        }

        # When
        kept_indexes = [0, 2, 4]
        featurizer.tfidf_vectorizer.fit_transform(utterances)
        self.assertDictEqual(voca, featurizer.tfidf_vectorizer.vocabulary_)
        diag = featurizer.tfidf_vectorizer.idf_.copy()
        featurizer._limit_tfidf_vectorizer_vocabulary(kept_indexes)

        # Then
        expected_voca = {
            "5": 0,
            "6": 1,
            "666": 2
        }
        self.assertDictEqual(
            expected_voca, featurizer.tfidf_vectorizer.vocabulary_)

        expected_diag = diag[kept_indexes].tolist()
        self.assertListEqual(
            expected_diag, featurizer.tfidf_vectorizer.idf_.tolist())

    def test_fit_transform_should_be_consistent_with_transform(self):
        # Here we mainly test that the output of fit_transform is
        # the same as the result of fit and then transform.
        # We're trying to avoid that for some reason indexes of features
        # get mixed up after feature selection

        # Given
        config = FeaturizerConfig(added_cooccurrence_feature_ratio=.5)
        featurizer = Featurizer(config=config)

        dataset = {
            "intents": {
                "intent1": {
                    "utterances": []
                }
            },
            "entities": {
                "entity_1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["alternative entity 1"]
                        },
                        {
                            "value": "éntity 1",
                            "synonyms": ["alternative entity 1"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": False,
                    "matching_strictness": 1.0
                },
                "entity_2": {
                    "data": [
                        {
                            "value": "Éntity 2",
                            "synonyms": ["Éntity_2", "Alternative entity 2"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "matching_strictness": 1.0
                },
                "snips/number": {}
            },
            "language": "en",
        }

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
        x_0 = featurizer.fit_transform(dataset, utterances, classes)
        x_1 = featurizer.transform(utterances)

        # Then
        self.assertListEqual(x_0.todense().tolist(), x_1.todense().tolist())

    def test_fit_with_no_utterance_should_raise(self):
        # Given
        utterances = []
        classes = []
        dataset = SAMPLE_DATASET

        # When/Then
        with self.assertRaises(_EmptyDataError) as ctx:
            Featurizer().fit_transform(dataset, utterances, classes)

        self.assertEqual(
            "Couldn't fit because no utterance was found an",
            str(ctx.exception))

    def test_limit_tfidf_vectorizer_vocabulary_should_raise(self):
        # Given
        featurizer = Featurizer()
        featurizer.tfidf_vectorizer = _get_tfidf_vectorizer("en")
        voca = {
            ("5",): 0,
            ("55",): 1,
            ("6",): 2,
            ("66",): 3,
            ("666",): 4
        }
        featurizer.tfidf_vectorizer.vocabulary_ = voca
        diag = [1.1 for _ in voca]

        num_features = len(voca)
        featurizer.tfidf_vectorizer._tfidf._idf_diag = spdiags(
            diag, diags=0, m=num_features, n=num_features, format="csr")

        # When / Then
        kept_indexes = [8, 9]

        with self.assertRaises(ValueError) as ctx:
            featurizer._limit_tfidf_vectorizer_vocabulary(kept_indexes)

        expected_ctx = "Invalid ngrams indexes set([8, 9]), expected values" \
                       " in set([0, 1, 2, 3, 4])"
        self.assertEqual(expected_ctx, str(ctx.exception))

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

        vectorizer_path = self.tmp_file_path / "cooccurrence_vectorizer.json"
        expected_vectorizer = {
            "word_pairs": {
                "0": ["yo", "yo"]
            },
            "language": "en",
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
            "language": "en",
            "word_pairs": serializable_word_pairs,
            "config": config.to_dict(),
        }

        self.tmp_file_path.mkdir()
        self.writeJsonContent(
            self.tmp_file_path / "cooccurrence_vectorizer.json",
            vectorizer_dict
        )

        # When
        vectorizer = CooccurrenceVectorizer.from_path(self.tmp_file_path)

        # Then
        self.assertDictEqual(config.to_dict(), vectorizer.config.to_dict())
        self.assertEqual("en", vectorizer.language)
        self.assertDictEqual(vectorizer.word_pairs, word_pairs)

    def test_cooccurence_vectorizer_should_preprocess(self):
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
        vectorizer.language = "en"

        # When
        preprocessed = vectorizer._preprocess(x)

        # Then
        expected = [["a", "b", "THE_C_ENTITY", "d", "THE_SNIPS_E_ENTITY", "f"]]
        self.assertSequenceEqual(expected, preprocessed)

    def test_cooccurence_vectorizer_should_transform(self):
        # Given
        config = CooccurrenceVectorizerConfig(
            use_stop_words=True,
            window_size=3,
            unknown_words_replacement_string="d")
        vectorizer = CooccurrenceVectorizer(config)
        vectorizer.language = "en"
        vectorizer.word_pairs = {
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

    def test_cooccurence_vectorizer_should_fit(self):
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

    def test_cooccurence_vectorizer_should_limit_vocabulary(self):
        # Given
        config = CooccurrenceVectorizerConfig(use_stop_words=False)
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
        self.assertDictEqual(pairs, vectorizer.word_pairs)

        # When
        kept_pairs_indexes = [0, 1, 2]
        vectorizer.limit_vocabulary(kept_pairs_indexes)

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

    def test_cooccurence_vectorizer_limit_vocabulary_should_raise(self):
        # Given
        vectorizer = CooccurrenceVectorizer()
        vectorizer.word_pairs = {
            ("a", "b"): 0,
            ("a", "c"): 1,
            ("a", "d"): 2,
            ("a", "e"): 3
        }

        # When / Then
        with self.assertRaises(ValueError) as ctx:
            vectorizer.limit_vocabulary([0, 8, 10])

        self.assertEqual(
            str(ctx.exception),
            "Invalid word pairs indexes set([8, 10]), expected values in "
            "set([0, 1, 2, 3])"
        )
