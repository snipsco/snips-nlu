# coding=utf-8
from __future__ import unicode_literals

import json

import numpy as np
from future.builtins import bytes
from future.utils import iteritems
from mock import patch, mock

from snips_nlu.constants import LANGUAGE_EN, DATA, TEXT, ENTITY, SLOT_NAME
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier.featurizer import (
    Featurizer, _get_tfidf_vectorizer, _get_utterances_to_features_names)
from snips_nlu.intent_classifier.log_reg_classifier_utils import \
    text_to_utterance
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.tests.utils import SnipsTest
from snips_nlu.tokenization import tokenize_light


class TestIntentClassifierFeaturizer(SnipsTest):
    def test_should_be_serializable(self):
        # Given
        language = LANGUAGE_EN
        tfidf_vectorizer = _get_tfidf_vectorizer(language)

        pvalue_threshold = 0.42
        featurizer = Featurizer(
            language,
            config=FeaturizerConfig(pvalue_threshold=pvalue_threshold,
                                    word_clusters_name="brown_clusters"),
            unknown_words_replacement_string=None,
            tfidf_vectorizer=tfidf_vectorizer)
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
                    "automatically_extensible": True
                }
            },
            "intents": {},
            "snips_nlu_version": "1.0.1",
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
            dumped = bytes(json.dumps(serialized_featurizer),
                           encoding="utf8").decode("utf8")

        msg = "SnipsNLUEngine should be deserializable from dict with unicode" \
              " values"
        with self.fail_if_exception(msg):
            _ = Featurizer.from_dict(json.loads(dumped))

        vocabulary = tfidf_vectorizer.vocabulary_
        # pylint: disable=W0212
        idf_diag = tfidf_vectorizer._tfidf._idf_diag.data.tolist()
        # pylint: enable=W0212

        best_features = featurizer.best_features
        entity_utterances_to_feature_names = {
            "entity1": ["entityfeatureentity2"]
        }

        expected_serialized = {
            "config": {
                'sublinear_tf': False,
                'pvalue_threshold': pvalue_threshold,
                'word_clusters_name': "brown_clusters"
            },
            "language_code": "en",
            "tfidf_vectorizer": {"idf_diag": idf_diag, "vocab": vocabulary},
            "best_features": best_features,
            "entity_utterances_to_feature_names":
                entity_utterances_to_feature_names,
            "unknown_words_replacement_string": None
        }
        self.assertDictEqual(expected_serialized, serialized_featurizer)

    def test_should_be_deserializable(self):
        # Given
        language = LANGUAGE_EN
        idf_diag = [1.52, 1.21, 1.04]
        vocabulary = {"hello": 0, "beautiful": 1, "world": 2}

        best_features = [0, 1]
        config = {
            "pvalue_threshold": 0.4,
            "sublinear_tf": False,
            "word_clusters_name": "brown_clusters"
        }

        entity_utterances_to_feature_names = {
            "entity_1": ["entityfeatureentity_1"]
        }

        featurizer_dict = {
            "config": config,
            "language_code": language,
            "tfidf_vectorizer": {"idf_diag": idf_diag, "vocab": vocabulary},
            "best_features": best_features,
            "entity_utterances_to_feature_names":
                entity_utterances_to_feature_names,
            "unknown_words_replacement_string": None
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
        self.assertListEqual(featurizer.best_features, best_features)
        self.assertEqual(config, featurizer.config.to_dict())

        self.assertDictEqual(
            featurizer.entity_utterances_to_feature_names,
            {
                k: set(v) for k, v
                in iteritems(entity_utterances_to_feature_names)
            })

    @mock.patch("snips_nlu.dataset.get_string_variations")
    def test_get_utterances_entities(self, mocked_get_string_variations):
        # Given
        def mock_get_string_variations(variation, language):
            return {variation, variation.lower()}

        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": []

                }
            },
            "entities": {
                "entity1": {
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
                    "automatically_extensible": False
                },
                "entity2": {
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
                    "automatically_extensible": False
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }
        language = LANGUAGE_EN
        dataset = validate_and_format_dataset(dataset)

        # When
        utterance_to_feature_names = _get_utterances_to_features_names(
            dataset, language)

        # Then
        expected_utterance_to_entity_names = {
            "entity 1": {"entityfeatureentity2", "entityfeatureentity1"},
            "éntity 1": {"entityfeatureentity1"},
            "éntity 2": {"entityfeatureentity2"},
            "Éntity 2": {"entityfeatureentity2"},
            "Éntity_2": {"entityfeatureentity2"},
            "éntity_2": {"entityfeatureentity2"},
            "alternative entity 2": {"entityfeatureentity2"},
            "Alternative entity 2": {"entityfeatureentity2"}
        }

        self.assertDictEqual(
            utterance_to_feature_names, expected_utterance_to_entity_names)

    @patch("snips_nlu.intent_classifier.featurizer.get_word_clusters")
    @patch("snips_nlu.intent_classifier.featurizer.stem")
    def test_preprocess_utterances(self, mocked_stem, mocked_word_cluster):
        # Given
        language = LANGUAGE_EN

        def _stem(t):
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

        mocked_word_cluster.return_value = {
            "brown_clusters": {
                "beautiful": "cluster_1",
                "birdy": "cluster_2",
                "entity": "cluster_3"
            }
        }

        mocked_stem.side_effect = stem_function

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
                    "automatically_extensible": False
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
                    "automatically_extensible": False
                },
                "snips/number": {}
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }

        dataset = validate_and_format_dataset(dataset)

        utterances = [
            text_to_utterance("hÉllo wOrld Éntity_2"),
            text_to_utterance("beauTiful World entity 1"),
            text_to_utterance("Bird bïrdy"),
        ]

        labeled_utterance = {
            DATA: [
                {
                    TEXT: "beauTiful éntity "
                },
                {
                    TEXT: "1",
                    ENTITY: "snips/number",
                    SLOT_NAME: "number"
                },
                {
                    TEXT: " bIrd Éntity_2"
                }
            ]
        }
        utterances.append(labeled_utterance)
        labels = np.array([0, 0, 1, 1])

        featurizer = Featurizer(
            language,
            None,
            config=FeaturizerConfig(word_clusters_name="brown_clusters")
        ).fit(dataset, utterances, labels)

        # When
        utterances = featurizer.preprocess_utterances(utterances)

        # Then
        expected_utterances = [
            "hello world entity_2 builtinentityfeaturesnipsnumber "
            "entityfeatureentity_2",
            "beauty world ent 1 builtinentityfeaturesnipsnumber "
            "entityfeatureentity_1 entityfeatureentity_2 "
            "cluster_1 cluster_3",
            "bird bird",
            "beauty ent bird entity_2 builtinentityfeaturesnipsnumber "
            "builtinentityfeaturesnipsnumber entityfeatureentity_1 "
            "entityfeatureentity_2 entityfeatureentity_2 cluster_1"
        ]

        self.assertListEqual(utterances, expected_utterances)

    def test_featurizer_should_exclude_replacement_string(self):
        # Given
        language = LANGUAGE_EN
        dataset = {
            "entities": {
                "dummy1": {
                    "utterances": {
                        "unknownword": "unknownword",
                        "what": "what"
                    }
                }
            }
        }
        replacement_string = "unknownword"
        featurizer = Featurizer(
            language, unknown_words_replacement_string=replacement_string,
            config=FeaturizerConfig())
        utterances = [text_to_utterance("hello dude")]
        y = np.array([1])

        # When
        featurizer.fit(dataset, utterances, y)

        # Then
        self.assertNotIn(replacement_string,
                         featurizer.entity_utterances_to_feature_names)

    def test_featurizer_should_be_serialized_when_not_fitted(self):
        # Given
        language = LANGUAGE_EN
        featurizer = Featurizer(language, None)
        # When/Then
        featurizer.to_dict()
