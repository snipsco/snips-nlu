from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from snips_nlu.languages import Language
from snips_nlu.resources import get_word_clusters
from snips_nlu.tokenization import tokenize_light, tokenize


def default_tfidf_vectorizer(language):
    return TfidfVectorizer(
        tokenizer=lambda x: [t.value for t in tokenize(x, language)])


def default_word_clusters_tfidf_vectorizer():
    return TfidfVectorizer(tokenizer=lambda x: x.split())


def get_tokens_clusters(tokens, language, cluster_name):
    return [get_word_clusters(language)[cluster_name][t] for t in tokens
            if t in get_word_clusters(language)[cluster_name]]


CLUSTER_USED_PER_LANGUAGES = {
    Language.ZH: "brown_clusters"
}


def deserialize_tfidf_vectorizer(vectorizer_dict, language):
    tfidf_vectorizer = default_tfidf_vectorizer(language)
    tfidf_vectorizer.vocabulary_ = vectorizer_dict["vocab"]
    idf_diag_data = np.array(vectorizer_dict["idf_diag"])
    idf_diag_shape = (len(idf_diag_data), len(idf_diag_data))
    row = range(idf_diag_shape[0])
    col = range(idf_diag_shape[0])
    idf_diag = sp.csr_matrix((idf_diag_data, (row, col)), shape=idf_diag_shape)
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer._idf_diag = idf_diag
    tfidf_vectorizer._tfidf = tfidf_transformer
    return tfidf_vectorizer


class Featurizer(object):
    def __init__(self, language, tfidf_vectorizer=None,
                 word_clusters_tfidf_vectorizer=None, pvalue_threshold=0.4):
        self.language = language
        if tfidf_vectorizer is None:
            tfidf_vectorizer = default_tfidf_vectorizer(self.language)
        self.tfidf_vectorizer = tfidf_vectorizer

        if language in CLUSTER_USED_PER_LANGUAGES:
            if word_clusters_tfidf_vectorizer is None:
                word_clusters_tfidf_vectorizer = \
                    default_word_clusters_tfidf_vectorizer()
            self.word_clusters_tfidf_vectorizer = \
                word_clusters_tfidf_vectorizer
        else:
            self.word_clusters_tfidf_vectorizer = None
        self.pvalue_threshold = pvalue_threshold
        self.best_features = None

    def fit(self, queries, y):
        if all(len("".join(tokenize_light(q, self.language))) == 0
               for q in queries):
            return None
        X = self.tfidf_vectorizer.fit_transform(query.encode('utf-8')
                                                for query in queries)

        if self.word_clusters_tfidf_vectorizer is not None:
            tokenized_queries = [tokenize_light(q, self.language)
                                 for q in queries]
            queries_clusters = [
                " ".join(get_tokens_clusters(
                    q, self.language,
                    CLUSTER_USED_PER_LANGUAGES[self.language]))
                for q in tokenized_queries]
            if all(len(q) == 0 for q in queries_clusters):
                self.word_clusters_tfidf_vectorizer = None
            else:
                X_cluster = self.word_clusters_tfidf_vectorizer \
                    .fit_transform(queries_clusters)
                X = sp.hstack((X, X_cluster), format="csr")

        chi2val, pval = chi2(X, y)

        self.best_features = [i for i, v in enumerate(pval) if
                              v < self.pvalue_threshold]
        if len(self.best_features) == 0:
            self.best_features = [idx for idx, val in enumerate(pval) if
                                  val == pval.min()]

        return self

    def transform(self, queries):
        X = self.tfidf_vectorizer.transform(queries)
        if self.word_clusters_tfidf_vectorizer is not None:
            tokenized_queries = [tokenize_light(q, self.language)
                                 for q in queries]
            queries_clusters = [
                " ".join(get_tokens_clusters(
                    q, self.language,
                    CLUSTER_USED_PER_LANGUAGES[self.language]))
                for q in tokenized_queries]
            X_cluster_tfidf = self.word_clusters_tfidf_vectorizer.transform(
                queries_clusters)
            X = sp.hstack((X, X_cluster_tfidf), format="csr")
        X = X[:, self.best_features]
        return X

    def fit_transform(self, queries, y):
        return self.fit(queries, y).transform(queries)

    def to_dict(self):
        tfidf_vectorizer = {
            'vocab': self.tfidf_vectorizer.vocabulary_,
            'idf_diag': self.tfidf_vectorizer._tfidf._idf_diag.data.tolist()
        }

        if self.word_clusters_tfidf_vectorizer is not None:
            word_clusters_tfidf_vectorizer = {
                'vocab': self.word_clusters_tfidf_vectorizer.vocabulary_,
                'idf_diag':
                    self.word_clusters_tfidf_vectorizer.
                        _tfidf._idf_diag.data.tolist(),
            }
        else:
            word_clusters_tfidf_vectorizer = None

        return {
            'language_code': self.language.iso_code,
            'tfidf_vectorizer': tfidf_vectorizer,
            'word_clusters_tfidf_vectorizer': word_clusters_tfidf_vectorizer,
            'pvalue_threshold': self.pvalue_threshold,
            'best_features': self.best_features,
        }

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict['language_code'])
        tfidf_vectorizer = deserialize_tfidf_vectorizer(
            obj_dict["tfidf_vectorizer"], language)
        if obj_dict["word_clusters_tfidf_vectorizer"] is not None:
            word_clusters_tfidf_vectorizer = deserialize_tfidf_vectorizer(
                vectorizer_dict=obj_dict["word_clusters_tfidf_vectorizer"],
                language=language)
        else:
            word_clusters_tfidf_vectorizer = None
        self = cls(
            language=language,
            tfidf_vectorizer=tfidf_vectorizer,
            word_clusters_tfidf_vectorizer=word_clusters_tfidf_vectorizer,
            pvalue_threshold=obj_dict['pvalue_threshold']
        )
        self.best_features = obj_dict["best_features"]
        return self
