from __future__ import unicode_literals

import json
import traceback as tb
import unittest

from mock import patch

from snips_nlu.intent_classifier.feature_extraction import (
    Featurizer, default_tfidf_vectorizer,
    default_word_clusters_tfidf_vectorizer)
from snips_nlu.languages import Language


class TestFeatureExtraction(unittest.TestCase):
    @patch("snips_nlu.intent_classifier.feature_extraction."
           "CLUSTER_USED_PER_LANGUAGES", {Language.EN: "brown_clusters"})
    def test_should_be_serializable(self):
        # Given
        language = Language.EN
        tfidf_vectorizer = default_tfidf_vectorizer(language)
        word_clusters_tfidf_vectorizer = \
            default_word_clusters_tfidf_vectorizer()
        pvalue_threshold = 0.42
        featurizer = Featurizer(
            language, tfidf_vectorizer=tfidf_vectorizer,
            word_clusters_tfidf_vectorizer=word_clusters_tfidf_vectorizer,
            pvalue_threshold=pvalue_threshold)

        queries = [
            "hello world",
            "beautiful world",
            "hello here",
            "bird birdy",
            "beautiful bird"
        ]
        classes = [0, 0, 0, 1, 1]

        featurizer.fit(queries, classes)

        # When
        serialized_featurizer = featurizer.to_dict()

        # Then
        try:
            dumped = json.dumps(serialized_featurizer).decode("utf8")
        except:
            self.fail("Featurizer dict should be json serializable to utf8.\n"
                      "Traceback:\n%s" % tb.format_exc())

        try:
            _ = Featurizer.from_dict(json.loads(dumped))
        except:
            self.fail("SnipsNLUEngine should be deserializable from dict with "
                      "unicode values\nTraceback:\n%s" % tb.format_exc())

        vocabulary = tfidf_vectorizer.vocabulary_
        idf_diag = tfidf_vectorizer._tfidf._idf_diag.data.tolist()

        word_clusters_vocabulary = word_clusters_tfidf_vectorizer.vocabulary_
        word_clusters_idf_diag = word_clusters_tfidf_vectorizer._tfidf \
            ._idf_diag.data.tolist()

        best_features = featurizer.best_features
        expected_serialized = {
            "language_code": "en",
            "tfidf_vectorizer": {"idf_diag": idf_diag, "vocab": vocabulary},
            "word_clusters_tfidf_vectorizer": {
                "idf_diag": word_clusters_idf_diag,
                "vocab": word_clusters_vocabulary
            },
            "best_features": best_features,
            "pvalue_threshold": pvalue_threshold
        }
        self.assertDictEqual(expected_serialized, serialized_featurizer)

    @patch("snips_nlu.intent_classifier.feature_extraction."
           "CLUSTER_USED_PER_LANGUAGES", {Language.EN: "brown_clusters"})
    def test_should_be_deserializable(self):
        # Given
        language = Language.EN
        idf_diag = [1.52, 1.21, 1.04]
        vocabulary = {"hello": 0, "beautiful": 1, "world": 2}

        word_clusters_idf_diag = [2, 4, 6]
        word_clusters_vocabulary = {"11101": 0, "101": 1, "111011": 2}

        best_features = [0, 1]
        pvalue_threshold = 0.4

        featurizer_dict = {
            "language_code": language.iso_code,
            "tfidf_vectorizer": {"idf_diag": idf_diag, "vocab": vocabulary},
            "word_clusters_tfidf_vectorizer": {
                "idf_diag": word_clusters_idf_diag,
                "vocab": word_clusters_vocabulary
            },
            "best_features": best_features,
            "pvalue_threshold": pvalue_threshold

        }

        # When
        featurizer = Featurizer.from_dict(featurizer_dict)

        # Then
        self.assertEqual(featurizer.language, language)
        self.assertListEqual(
            featurizer.tfidf_vectorizer._tfidf._idf_diag.data.tolist(),
            idf_diag)
        self.assertDictEqual(featurizer.tfidf_vectorizer.vocabulary_,
                             vocabulary)
        self.assertListEqual(featurizer.best_features, best_features)
        self.assertEqual(featurizer.pvalue_threshold, pvalue_threshold)
