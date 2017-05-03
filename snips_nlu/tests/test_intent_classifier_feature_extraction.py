from __future__ import unicode_literals

import json
import unittest
from contextlib import contextmanager
from copy import deepcopy

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import snips_nlu
from snips_nlu.intent_classifier.feature_extraction import Featurizer, \
    process_builtin_entities
from snips_nlu.languages import Language
from snips_nlu.utils import safe_pickle_dumps

WORD_CLUSTERS = {}


@contextmanager
def set_empty_word_cluster():
    mod = snips_nlu.intent_classifier.feature_extraction
    old_value = deepcopy(getattr(mod, "WORD_CLUSTERS"))
    setattr(mod, "WORD_CLUSTERS", {})
    yield
    setattr(mod, "WORD_CLUSTERS", old_value)


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

        count_vectorizer_pickled = safe_pickle_dumps(count_vectorizer)
        tfidf_transformer_pickled = safe_pickle_dumps(tfidf_transformer)
        cluster_count_vectorizer_pickled = safe_pickle_dumps(
            featurizer.word_cluster_count_vectorizer)
        cluster_tfidf_transformer_pickled = safe_pickle_dumps(
            featurizer.word_cluster_tfidf_transformer)
        length_scaler_pickled = safe_pickle_dumps(featurizer.length_scaler)

        expected_serialized = {
            "language_code": "en",
            "count_vectorizer": count_vectorizer_pickled,
            "tfidf_transformer": tfidf_transformer_pickled,
            "word_cluster_count_vectorizer": cluster_count_vectorizer_pickled,
            "word_cluster_tfidf_transformer": cluster_tfidf_transformer_pickled,
            "length_scaler": length_scaler_pickled,
            "best_features": None,
            "pvalue_threshold": 0.4
        }
        self.assertDictEqual(expected_serialized, serialized_featurizer)

    def test_process_builtin_entities(self):
        # Given
        text = "hey dude let's meet at 8 p.m, what do you think?"
        language = Language.EN

        # When
        new_text = process_builtin_entities(text, language)

        # Then
        expected_new_text = "snipsdatetime hey dude let s meet what do you " \
                            "think ?"
        self.assertEqual(expected_new_text, new_text)

    def test_should_add_builtin_entities_features(self):
        # Given
        texts = ["hey dude let's meet at 8 p.m, what do you think?",
                 "there is nothing here"]
        language = Language.EN
        featutizer = Featurizer(language, pvalue_threshold=float("inf"))

        # When
        labels = [1, 0]
        feature_1 = featutizer.fit_transform(texts, labels)

        with set_empty_word_cluster():
            feature_2 = featutizer.fit_transform(texts, labels)

        # Then
        self.assertTrue(feature_1.shape[1] > feature_2.shape[1])
