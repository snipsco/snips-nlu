from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from snips_nlu.languages import Language
from snips_nlu.resources import get_stop_words
from snips_nlu.tokenization import tokenize_light, tokenize


def default_tfidf_vectorizer(language):
    return TfidfVectorizer(
        tokenizer=lambda x: [t.value for t in tokenize(x, language)])


class Featurizer(object):
    def __init__(self, language, tfidf_vectorizer=None, pvalue_threshold=0.4):
        self.language = language
        if tfidf_vectorizer is None:
            tfidf_vectorizer = default_tfidf_vectorizer(self.language)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.best_features = None
        self.pvalue_threshold = pvalue_threshold

    def fit(self, queries, y):
        if all(len("".join(tokenize_light(q, self.language))) == 0
               for q in queries):
            return None
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(
            query.encode('utf-8') for query in queries)
        list_index_words = {self.tfidf_vectorizer.vocabulary_[x]: x for x in
                            self.tfidf_vectorizer.vocabulary_}

        stop_words = get_stop_words(self.language)

        chi2val, pval = chi2(X_train_tfidf, y)
        self.best_features = [i for i, v in enumerate(pval) if
                              v < self.pvalue_threshold]
        if len(self.best_features) == 0:
            self.best_features = [idx for idx, val in enumerate(pval) if
                                  val == pval.min()]

        feature_names = {}
        for i in self.best_features:
            feature_names[i] = {'word': list_index_words[i], 'pval': pval[i]}

        for feat in feature_names:
            if feature_names[feat]['word'] in stop_words:
                if feature_names[feat]['pval'] > self.pvalue_threshold / 2.0:
                    self.best_features.remove(feat)

        return self

    def transform(self, queries):
        X_train_tfidf = self.tfidf_vectorizer.transform(queries)
        X = X_train_tfidf[:, self.best_features]
        return X

    def fit_transform(self, queries, y):
        return self.fit(queries, y).transform(queries)

    def to_dict(self):
        idf_diag = self.tfidf_vectorizer._tfidf._idf_diag.data.tolist()
        return {
            'language_code': self.language.iso_code,
            'tfidf_vectorizer_vocab': self.tfidf_vectorizer.vocabulary_,
            'tfidf_vectorizer_stop_words': self.tfidf_vectorizer.stop_words,
            'tfidf_vectorizer_idf_diag': idf_diag,
            'best_features': self.best_features,
            'pvalue_threshold': self.pvalue_threshold
        }

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict['language_code'])
        tfidf_vectorizer = default_tfidf_vectorizer(language)
        tfidf_vectorizer.vocabulary_ = obj_dict['tfidf_vectorizer_vocab']
        tfidf_vectorizer.stop_words = obj_dict['tfidf_vectorizer_stop_words']
        idf_diag_data = np.array(obj_dict['tfidf_vectorizer_idf_diag'])
        idf_diag_shape = (len(idf_diag_data), len(idf_diag_data))
        row = range(idf_diag_shape[0])
        col = range(idf_diag_shape[0])
        idf_diag = sp.csr_matrix((idf_diag_data, (row, col)),
                                 shape=idf_diag_shape)
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer._idf_diag = idf_diag
        tfidf_vectorizer._tfidf = tfidf_transformer
        self = cls(
            language=language,
            tfidf_vectorizer=tfidf_vectorizer,
            pvalue_threshold=obj_dict['pvalue_threshold']
        )
        self.best_features = obj_dict["best_features"]
        return self
