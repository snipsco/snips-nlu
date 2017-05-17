from __future__ import unicode_literals

import unittest

from sklearn.feature_extraction.text import TfidfVectorizer

from snips_nlu.intent_classifier.feature_extraction import Featurizer
from snips_nlu.languages import Language
from snips_nlu.tokenization import tokenize_light


class TestFeaturizer(unittest.TestCase):
    def test_should_be_saveable(self):
        # Given
        language = Language.EN
        pvalue_threshold = 0.51
        stop_words = ["the", "is"]
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_light,
                                           stop_words=stop_words)
        featurizer = Featurizer(language, tfidf_vectorizer, pvalue_threshold)
        queries = [
            "hello beautiful world",
            "the world is nice",
            "awful world",
            "hello bird",
            "nice bird",
            "blue bird"
        ]

        classes = [0, 0, 0, 1, 1, 1]
        featurizer.fit(queries, classes)

        # When
        featurizer_dict = featurizer.to_dict()

        # Then
        expected_vocab = {
            'awful': 0,
            'beautiful': 1,
            'bird': 2,
            'blue': 3,
            'hello': 4,
            'nice': 5,
            'world': 6
        }
        self.assertEqual(featurizer_dict['language_code'], language.iso_code)
        self.assertEqual(featurizer_dict['pvalue_threshold'], pvalue_threshold)
        self.assertDictEqual(featurizer_dict['tfidf_vectorizer_vocab'],
                             expected_vocab)
        self.assertListEqual(featurizer_dict['tfidf_vectorizer_stop_words'],
                             stop_words)
        self.assertTrue(isinstance(featurizer_dict['best_features'], list))
        self.assertTrue(
            isinstance(featurizer_dict['tfidf_vectorizer_idf_diag'], list))

    def test_should_be_loadable(self):
        # Given
        pvalue_threshold = 0.51
        best_features = [0, 1, 2, 3, 6]
        vocabulary = {
            'awful': 0,
            'beautiful': 1,
            'bird': 2,
            'blue': 3,
            'hello': 4,
            'nice': 5,
            'world': 6
        }
        idf_diag = [2.252762968495368, 2.252762968495368, 1.5596157879354227,
                    2.252762968495368, 1.8472978603872037, 1.8472978603872037,
                    1.5596157879354227]
        stop_words = ['is', 'the']
        language = Language.EN
        featurizer_dict = {
            'tfidf_vectorizer_idf_diag': idf_diag,
            'pvalue_threshold': pvalue_threshold,
            'best_features': best_features,
            'tfidf_vectorizer_vocab': vocabulary,
            'tfidf_vectorizer_stop_words': stop_words,
            'language_code': language.iso_code
        }

        # When
        featurizer = Featurizer.from_dict(featurizer_dict)

        # Then
        self.assertEqual(featurizer.language, language)
        self.assertEqual(featurizer.pvalue_threshold, pvalue_threshold)
        self.assertListEqual(featurizer.best_features, best_features)
        self.assertListEqual(featurizer.tfidf_vectorizer.stop_words,
                             stop_words)
        self.assertDictEqual(featurizer.tfidf_vectorizer.vocabulary_,
                             vocabulary)
        self.assertListEqual(
            featurizer.tfidf_vectorizer._tfidf._idf_diag.data.tolist(),
            idf_diag)
