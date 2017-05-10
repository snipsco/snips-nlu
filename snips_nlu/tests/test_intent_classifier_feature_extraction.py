import json
import unittest

from snips_nlu.intent_classifier.feature_extraction import (
    Featurizer, default_tfidf_vectorizer)
from snips_nlu.languages import Language


class TestFeatureExtraction(unittest.TestCase):
    def test_should_be_serializable(self):
        # Given
        language = Language.EN
        tfidf_vectorizer = default_tfidf_vectorizer()
        pvalue_threshold = 0.42
        featurizer = Featurizer(language, tfidf_vectorizer=tfidf_vectorizer,
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
        # noinspection PyBroadException
        try:
            dumped = json.dumps(serialized_featurizer).decode("utf8")
        except:
            self.fail("Featurizer dict should be json serializable to utf8")

        # noinspection PyBroadException
        try:
            _ = Featurizer.from_dict(json.loads(dumped))
        except:
            self.fail("SnipsNLUEngine should be deserializable from dict with "
                      "unicode values")

        stop_words = tfidf_vectorizer.stop_words
        vocabulary = tfidf_vectorizer.vocabulary_
        idf_diag = tfidf_vectorizer._tfidf._idf_diag.data.tolist()
        best_features = featurizer.best_features
        expected_serialized = {
            "language_code": "en",
            "tfidf_vectorizer_idf_diag": idf_diag,
            "tfidf_vectorizer_stop_words": stop_words,
            "tfidf_vectorizer_vocab": vocabulary,
            "best_features": best_features,
            "pvalue_threshold": pvalue_threshold
        }
        self.assertDictEqual(expected_serialized, serialized_featurizer)

    def test_should_be_deserializable(self):
        # Given
        language = Language.EN
        idf_diag = [1.52, 1.21, 1.04]
        stop_words = ["the", "at"]
        vocabulary = {"hello": 0, "beautiful": 1, "world": 2}
        best_features = [0, 1]
        pvalue_threshold = 0.4

        featurizer_dict = {
            "language_code": language.iso_code,
            "tfidf_vectorizer_idf_diag": idf_diag,
            "tfidf_vectorizer_stop_words": stop_words,
            "tfidf_vectorizer_vocab": vocabulary,
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
        self.assertListEqual(featurizer.tfidf_vectorizer.stop_words,
                             stop_words)
        self.assertDictEqual(featurizer.tfidf_vectorizer.vocabulary_,
                             vocabulary)
        self.assertListEqual(featurizer.best_features, best_features)
        self.assertEqual(featurizer.pvalue_threshold, pvalue_threshold)
