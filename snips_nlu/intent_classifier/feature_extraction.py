from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from snips_nlu.constants import (NGRAM)
from snips_nlu.languages import Language
from snips_nlu.resources import get_stop_words, get_word_clusters
from snips_nlu.slot_filler.features_utils import get_all_ngrams
from snips_nlu.tokenization import tokenize_light


def default_tfidf_vectorizer(language):
    return TfidfVectorizer(tokenizer=lambda x: tokenize_light(x, language))


def get_tokens_clusters(tokens, language, cluster_name):
    return [get_word_clusters(language)[cluster_name][t] for t in tokens
            if t in get_word_clusters(language)[cluster_name]]


def entity_name_to_feature(entity_name, language):
    return "entityfeature%s" % "".join(tokenize_light(
        entity_name, language=language))


def add_word_cluster_features_to_query(query, language):
    cluster_name = CLUSTER_USED_PER_LANGUAGES.get(language, False)
    if not cluster_name:
        return query
    tokens = tokenize_light(query, language)
    ngrams = get_all_ngrams(tokens)
    match_features = []
    for ngram in ngrams:
        cluster = get_word_clusters(language)[cluster_name].get(
            ngram[NGRAM], False)
        if cluster:
            match_features.append(cluster)
    if len(match_features) > 0:
        query += " " + " ".join(match_features)
    return query


def preprocess_query(query, language):
    return add_word_cluster_features_to_query(query, language)


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


CLUSTER_USED_PER_LANGUAGES = {}


class Featurizer(object):
    def __init__(self, language, tfidf_vectorizer=None, pvalue_threshold=0.4):
        self.language = language
        if tfidf_vectorizer is None:
            tfidf_vectorizer = default_tfidf_vectorizer(self.language)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.pvalue_threshold = pvalue_threshold
        self.best_features = None

    def preprocess_queries(self, queries):
        preprocessed_queries = []
        for q in queries:
            processed_query = preprocess_query(q, self.language)
            processed_query = processed_query.encode("utf8")
            preprocessed_queries.append(processed_query)
        return preprocessed_queries

    def fit(self, queries, y):
        if all(len("".join(tokenize_light(q, self.language))) == 0
               for q in queries):
            return None

        preprocessed_queries = self.preprocess_queries(queries)

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(
            preprocessed_queries)

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
        preprocessed_queries = self.preprocess_queries(queries)
        X_train_tfidf = self.tfidf_vectorizer.transform(preprocessed_queries)
        X = X_train_tfidf[:, self.best_features]
        return X

    def fit_transform(self, queries, y):
        return self.fit(queries, y).transform(queries)

    def to_dict(self):
        tfidf_vectorizer = {
            'vocab': self.tfidf_vectorizer.vocabulary_,
            'idf_diag': self.tfidf_vectorizer._tfidf._idf_diag.data.tolist()
        }

        return {
            'language_code': self.language.iso_code,
            'tfidf_vectorizer': tfidf_vectorizer,
            'pvalue_threshold': self.pvalue_threshold,
            'best_features': self.best_features,
        }

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict['language_code'])
        tfidf_vectorizer = deserialize_tfidf_vectorizer(
            obj_dict["tfidf_vectorizer"], language)
        self = cls(
            language=language,
            tfidf_vectorizer=tfidf_vectorizer,
            pvalue_threshold=obj_dict['pvalue_threshold']
        )
        self.best_features = obj_dict["best_features"]
        return self
