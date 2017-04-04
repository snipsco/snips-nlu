import cPickle
import json
import unittest

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from snips_nlu.intent_classifier.feature_extraction import Featurizer
from snips_nlu.languages import Language


class TestFeatureExtraction(unittest.TestCase):
    def test_should_be_serializable(self):
        # Given
        language = Language.EN
        count_vectorizer = CountVectorizer(ngram_range=(1, 1))

        tfidf_transformer = TfidfTransformer()
        featurizer = Featurizer(language, count_vectorizer=count_vectorizer,
                                tfidf_transformer=tfidf_transformer)

        # When
        serialized_featurizer = featurizer.to_dict()

        # Then
        try:
            dumped = json.dumps(serialized_featurizer).decode("utf8")
        except:
            self.fail("Featurizer dict should be json serializable to utf8")

        try:
            _ = Featurizer.from_dict(json.loads(dumped))
        except:
            self.fail("SnipsNLUEngine should be deserializable from dict with "
                      "unicode values")

        count_vectorizer_pickled = cPickle.dumps(count_vectorizer)
        tfidf_transformer_pickled = cPickle.dumps(tfidf_transformer)
        expected_serialized = {
            "language_code": "en",
            "count_vectorizer": count_vectorizer_pickled,
            "tfidf_transformer": tfidf_transformer_pickled,
            "best_features": None,
            "pvalue_threshold": 0.4
        }
        self.assertDictEqual(expected_serialized, serialized_featurizer)
