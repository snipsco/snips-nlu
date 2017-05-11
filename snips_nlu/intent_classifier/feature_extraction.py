from __future__ import unicode_literals

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.preprocessing import RobustScaler

from snips_nlu.built_in_entities import get_built_in_entities
from snips_nlu.constants import MATCH_RANGE, ENTITY
from snips_nlu.languages import Language
from snips_nlu.resources import get_word_clusters
from snips_nlu.tokenization import tokenize_light, tokenize
from snips_nlu.utils import ensure_string, safe_pickle_dumps, safe_pickle_loads

WORD_CLUSTERS = {
    Language.EN: "brown_clusters"
}


def tokenize_and_normalize(text):
    return [t.lower() for t in tokenize_light(text)]


def default_count_vectorizer(tokenizer=tokenize_and_normalize):
    return CountVectorizer(ngram_range=(1, 1), tokenizer=tokenizer,
                           encoding="utf-8")


def default_tfidf_transformer():
    return TfidfTransformer()


def default_scaler():
    return RobustScaler(with_centering=True, with_scaling=True,
                        quantile_range=(25.0, 75.0), copy=True)


def get_tokens_clusters(tokens, language, cluster_name):
    return [get_word_clusters(language)[cluster_name][t] for t in tokens
            if t in get_word_clusters(language)[cluster_name]]


def normalize_entity_name(entity_name):
    return "".join(tokenize_light(entity_name))


def process_builtin_entities(text, language):
    built_in_entities = get_built_in_entities(text, language)
    text_tokens = tokenize(text)
    entity_tokens = set()
    processed_tokens = []
    for e in built_in_entities:
        processed_tokens.append(normalize_entity_name(e[ENTITY].label))
        for i, t in enumerate(text_tokens):
            if e[MATCH_RANGE][0] <= t.start < e[MATCH_RANGE][1] \
                    and e[MATCH_RANGE][0] < t.end <= e[MATCH_RANGE][1]:
                entity_tokens.add(i)
    processed_tokens += [t.value for i, t in enumerate(text_tokens)
                         if i not in entity_tokens]
    return " ".join(processed_tokens)


class Featurizer(object):
    def __init__(self, language, count_vectorizer=default_count_vectorizer(),
                 tfidf_transformer=default_tfidf_transformer(),
                 length_scaler=default_scaler(),
                 word_cluster_count_vectorizer=None,
                 word_cluster_tfidf_transformer=None, pvalue_threshold=0.4):
        self.language = language
        add_word_cluster = False
        try:
            get_word_clusters(language)
            add_word_cluster = True
        except KeyError:
            pass

        if add_word_cluster:
            if word_cluster_count_vectorizer is None:
                word_cluster_count_vectorizer = default_count_vectorizer()
            if word_cluster_tfidf_transformer is None:
                word_cluster_tfidf_transformer = default_tfidf_transformer()

        self.count_vectorizer = count_vectorizer
        self.word_cluster_count_vectorizer = word_cluster_count_vectorizer
        self.word_cluster_tfidf_transformer = word_cluster_tfidf_transformer
        self.tfidf_transformer = tfidf_transformer
        self.best_features = None
        self.pvalue_threshold = pvalue_threshold
        self.length_scaler = length_scaler

    def fit(self, queries, y):
        tokenized_queries = [tokenize_light(q) for q in queries]
        queries_lengths = [len(q) for q in tokenized_queries]
        X = np.array(queries_lengths, ndmin=2).reshape(
            (len(queries), -1))

        if self.language in WORD_CLUSTERS:
            cluster_queries = [
                " ".join(get_tokens_clusters(q, self.language,
                                             WORD_CLUSTERS[self.language]))
                for q in tokenized_queries]
            if all(len(q) == 0 for q in cluster_queries):
                self.word_cluster_count_vectorizer = None
                self.word_cluster_tfidf_transformer = None
            else:
                X_cluster_counts = self.word_cluster_count_vectorizer.fit_transform(
                    cluster_queries)
                X_cluster_tfidf = self.word_cluster_tfidf_transformer.fit_transform(
                    X_cluster_counts)
                X = np.concatenate((X, X_cluster_tfidf.todense()), axis=1)

        # Be careful here we transform the sentence, n-gram with n > 1 can't
        # be used as features anymore
        X_counts = self.count_vectorizer.fit_transform(
            process_builtin_entities(q, self.language).encode('utf-8')
            for q in queries)

        X_train_tfidf = self.tfidf_transformer.fit_transform(X_counts)
        X = np.concatenate((X, X_train_tfidf.todense()), axis=1)

        chi2val, pval = chi2(X, y)
        self.best_features = [i for i, v in enumerate(pval) if
                              v < self.pvalue_threshold]
        # Scale length
        X[:, 0] = self.length_scaler.fit_transform(X[:, 0])
        if len(self.best_features) == 0:
            self.best_features = [idx for idx, val in enumerate(pval) if
                                  val == pval.min()]
        return self

    def transform(self, queries):
        tokenized_queries = [tokenize_light(q) for q in queries]
        queries_lengths = [len(q) for q in tokenized_queries]
        queries_lengths = np.array(queries_lengths, ndmin=2).reshape(
            (len(queries), -1))
        X = self.length_scaler.transform(queries_lengths)

        if self.language in WORD_CLUSTERS \
                and self.word_cluster_count_vectorizer is not None:
            cluster_queries = [" ".join(get_tokens_clusters(
                q, self.language, WORD_CLUSTERS[self.language]))
                for q in tokenized_queries]
            X_cluster_counts = self.word_cluster_count_vectorizer.transform(
                cluster_queries)
            X_cluster_tfidf = self.word_cluster_tfidf_transformer.transform(
                X_cluster_counts)
            X = np.concatenate((X, X_cluster_tfidf.todense()), axis=1)

        X_counts = self.count_vectorizer.transform(
            process_builtin_entities(q, self.language).encode('utf-8')
            for q in queries)
        X_train_tfidf = self.tfidf_transformer.transform(X_counts)
        X = np.concatenate((X, X_train_tfidf.todense()), axis=1)
        return X[:, self.best_features]

    def fit_transform(self, queries, y):
        return self.fit(queries, y).transform(queries)

    def to_dict(self):
        return {
            'language_code': self.language.iso_code,
            'count_vectorizer': safe_pickle_dumps(self.count_vectorizer),
            'tfidf_transformer': safe_pickle_dumps(self.tfidf_transformer),
            'best_features': self.best_features,
            'pvalue_threshold': self.pvalue_threshold,
            'word_cluster_count_vectorizer': safe_pickle_dumps(
                self.word_cluster_count_vectorizer),
            'word_cluster_tfidf_transformer': safe_pickle_dumps(
                self.word_cluster_tfidf_transformer),
            'length_scaler': safe_pickle_dumps(self.length_scaler)
        }

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict['count_vectorizer'] = ensure_string(
            obj_dict['count_vectorizer'])
        obj_dict['tfidf_transformer'] = ensure_string(
            obj_dict['tfidf_transformer'])
        obj_dict["word_cluster_count_vectorizer"] = ensure_string(
            obj_dict["word_cluster_count_vectorizer"])
        obj_dict["word_cluster_tfidf_transformer"] = ensure_string(
            obj_dict["word_cluster_tfidf_transformer"])
        obj_dict["length_scaler"] = ensure_string(obj_dict["length_scaler"])
        self = cls(
            language=Language.from_iso_code(obj_dict['language_code']),
            count_vectorizer=safe_pickle_loads(obj_dict['count_vectorizer']),
            tfidf_transformer=safe_pickle_loads(obj_dict['tfidf_transformer']),
            pvalue_threshold=obj_dict['pvalue_threshold'],
            word_cluster_count_vectorizer=obj_dict[
                "word_cluster_count_vectorizer"],
            word_cluster_tfidf_transformer=obj_dict[
                "word_cluster_tfidf_transformer"],
            length_scaler=obj_dict["length_scaler"]
        )
        self.best_features = obj_dict["best_features"]
        return self
