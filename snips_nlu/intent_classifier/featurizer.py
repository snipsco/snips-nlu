from __future__ import division, unicode_literals

from builtins import object, range

import numpy as np
import scipy.sparse as sp
from future.utils import iteritems
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_is_fitted
from snips_nlu_utils import normalize

from snips_nlu.builtin_entities import (
    get_builtin_entity_parser, is_builtin_entity)
from snips_nlu.constants import (BUILTIN_ENTITY_PARSER, DATA, ENTITY,
                                 ENTITY_KIND, NGRAM, TEXT)
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.languages import get_default_sep
from snips_nlu.parser.custom_entity_parser import CustomEntityParser
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.preprocessing import stem, tokenize_light
from snips_nlu.resources import (
    MissingResource, get_stop_words, get_word_cluster)
from snips_nlu.slot_filler.features_utils import get_all_ngrams


class Featurizer(object):
    def __init__(self, language, unknown_words_replacement_string,
                 config=FeaturizerConfig(), tfidf_vectorizer=None,
                 best_features=None, entity_utterances_to_feature_names=None,
                 builtin_entity_parser=None, custom_entity_parser=None):
        self.config = config
        self.language = language
        if tfidf_vectorizer is None:
            tfidf_vectorizer = _get_tfidf_vectorizer(
                self.language, sublinear_tf=self.config.sublinear_tf)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.best_features = best_features
        self.entity_utterances_to_feature_names = \
            entity_utterances_to_feature_names

        self.unknown_words_replacement_string = \
            unknown_words_replacement_string

        self.builtin_entity_parser = builtin_entity_parser
        self.custom_entity_parser = custom_entity_parser

    def fit(self, dataset, utterances, classes):
        vectorizer_is_fitted = False
        try:
            check_is_fitted(self.tfidf_vectorizer, 'vocabulary_')
            vectorizer_is_fitted = True
        except NotFittedError:
            pass

        if self.builtin_entity_parser is None or vectorizer_is_fitted:
            self.builtin_entity_parser = get_builtin_entity_parser(dataset)

        if self.custom_entity_parser is None or vectorizer_is_fitted:
            self.custom_entity_parser = CustomEntityParser(None).fit(dataset)

        utterances_texts = (get_text_from_chunks(u[DATA]) for u in utterances)
        if not any(tokenize_light(q, self.language) for q in utterances_texts):
            return None

        preprocessed_utterances = self.preprocess_utterances(utterances)
        # pylint: disable=C0103
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(
            preprocessed_utterances)
        # pylint: enable=C0103
        features_idx = {self.tfidf_vectorizer.vocabulary_[word]: word for word
                        in self.tfidf_vectorizer.vocabulary_}

        stop_words = get_stop_words(self.language)

        _, pval = chi2(X_train_tfidf, classes)
        self.best_features = [i for i, v in enumerate(pval) if
                              v < self.config.pvalue_threshold]
        if not self.best_features:
            self.best_features = [idx for idx, val in enumerate(pval) if
                                  val == pval.min()]

        feature_names = {}
        for utterance_index in self.best_features:
            feature_names[utterance_index] = {
                "word": features_idx[utterance_index],
                "pval": pval[utterance_index]}

        for feat in feature_names:
            if feature_names[feat]["word"] in stop_words:
                if feature_names[feat]["pval"] > \
                        self.config.pvalue_threshold / 2.0:
                    self.best_features.remove(feat)

        return self

    def transform(self, utterances):
        preprocessed_utterances = self.preprocess_utterances(utterances)
        # pylint: disable=C0103
        X_train_tfidf = self.tfidf_vectorizer.transform(
            preprocessed_utterances)
        X = X_train_tfidf[:, self.best_features]
        # pylint: enable=C0103
        return X

    def fit_transform(self, dataset, queries, y):
        return self.fit(dataset, queries, y).transform(queries)

    def preprocess_utterances(self, utterances):
        return [
            _preprocess_utterance(
                u, self.language, self.builtin_entity_parser,
                self.entity_utterances_to_feature_names,
                self.config.word_clusters_name,
                self.unknown_words_replacement_string)
            for u in utterances
        ]

    def to_dict(self):
        """Returns a json-serializable dict"""
        if hasattr(self.tfidf_vectorizer, "vocabulary_"):
            # pylint: # pylint: disable=W0212
            vocab = {k: int(v) for k, v in
                     iteritems(self.tfidf_vectorizer.vocabulary_)}
            idf_diag = self.tfidf_vectorizer._tfidf._idf_diag.data.tolist()
            # pylint: enable=W0212
            entity_utterances_to_entity_names = {
                k: list(v)
                for k, v in iteritems(self.entity_utterances_to_feature_names)
            }
        else:
            vocab = None
            idf_diag = None
            entity_utterances_to_entity_names = dict()

        tfidf_vectorizer = {
            "vocab": vocab,
            "idf_diag": idf_diag
        }

        return {
            "language_code": self.language,
            "tfidf_vectorizer": tfidf_vectorizer,
            "best_features": self.best_features,
            "entity_utterances_to_feature_names":
                entity_utterances_to_entity_names,
            "config": self.config.to_dict(),
            "unknown_words_replacement_string":
                self.unknown_words_replacement_string
        }

    @classmethod
    def from_dict(cls, obj_dict, **shared):
        """Creates a :class:`Featurizer` instance from a :obj:`dict`

        The dict must have been generated with :func:`~Featurizer.to_dict`
        """
        language = obj_dict["language_code"]
        config = FeaturizerConfig.from_dict(obj_dict["config"])
        tfidf_vectorizer = _deserialize_tfidf_vectorizer(
            obj_dict["tfidf_vectorizer"], language, config.sublinear_tf)
        entity_utterances_to_entity_names = {
            k: set(v) for k, v in
            iteritems(obj_dict["entity_utterances_to_feature_names"])
        }
        self = cls(
            language=language,
            tfidf_vectorizer=tfidf_vectorizer,
            entity_utterances_to_feature_names=
            entity_utterances_to_entity_names,
            best_features=obj_dict["best_features"],
            config=config,
            unknown_words_replacement_string=obj_dict[
                "unknown_words_replacement_string"],
            builtin_entity_parser=shared.get(BUILTIN_ENTITY_PARSER)
        )
        return self


def _get_tfidf_vectorizer(language, sublinear_tf=False):
    return TfidfVectorizer(tokenizer=lambda x: tokenize_light(x, language),
                           sublinear_tf=sublinear_tf)


def _get_tokens_clusters(tokens, language, cluster_name):
    clusters = get_word_cluster(language, cluster_name)
    return [clusters[t] for t in tokens if t in clusters]


def _entity_name_to_feature(entity_name):
    # We use 'en' here since we tokenize the label and not the data
    return "entityfeature%s" % "".join(tokenize_light(
        entity_name, language="en"))


def _builtin_entity_to_feature(builtin_entity_label):
    # We use 'en' here since we tokenize the label and not the data
    return "builtinentityfeature%s" % "".join(tokenize_light(
        builtin_entity_label, language="en"))


def _normalize_stem(text, language):
    normalized_stemmed = normalize(text)
    try:
        normalized_stemmed = stem(normalized_stemmed, language)
    except MissingResource:
        pass
    return normalized_stemmed


def _get_word_cluster_features(query_tokens, clusters_name, language):
    if not clusters_name:
        return []
    ngrams = get_all_ngrams(query_tokens)
    cluster_features = []
    for ngram in ngrams:
        cluster = get_word_cluster(language, clusters_name).get(
            ngram[NGRAM].lower(), None)
        if cluster is not None:
            cluster_features.append(cluster)
    return cluster_features


def _preprocess_utterance(utterance, language, builtin_entity_parser,
                          custom_entity_parser,
                          word_clusters_name,
                          unknownword_replacement_string):
    utterance_text = get_text_from_chunks(utterance[DATA])
    utterance_tokens = tokenize_light(utterance_text, language)
    word_clusters_features = _get_word_cluster_features(
        utterance_tokens, word_clusters_name, language)
    normalized_stemmed_tokens = [_normalize_stem(t, language)
                                 for t in utterance_tokens]
    custom_entities = custom_entity_parser.parse(normalized_stemmed_tokens)
    custom_entities = [e for e in custom_entities
                       if e["raw_value"] != unknownword_replacement_string]
    entities_features = [_entity_name_to_feature(e["entity"])
                         for e in custom_entities]
    builtin_entities = builtin_entity_parser.parse(utterance_text,
                                                   use_cache=True)
    builtin_entities_features = [
        _builtin_entity_to_feature(ent[ENTITY_KIND]) for ent in
        builtin_entities
    ]

    # We remove values of builtin slots from the utterance to avoid learning
    # specific samples such as '42' or 'tomorrow'
    filtered_normalized_stemmed_tokens = [
        _normalize_stem(chunk[TEXT], language) for chunk in utterance[DATA]
        if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
    ]

    features = get_default_sep(language).join(
        filtered_normalized_stemmed_tokens)
    if builtin_entities_features:
        features += " " + " ".join(sorted(builtin_entities_features))
    if entities_features:
        features += " " + " ".join(sorted(entities_features))
    if word_clusters_features:
        features += " " + " ".join(sorted(word_clusters_features))

    return features


def _deserialize_tfidf_vectorizer(vectorizer_dict, language, sublinear_tf):
    tfidf_vectorizer = _get_tfidf_vectorizer(language, sublinear_tf)
    tfidf_transformer = TfidfTransformer()
    vocab = vectorizer_dict["vocab"]
    if vocab is not None:  # If the vectorizer has been fitted
        tfidf_vectorizer.vocabulary_ = vocab
        idf_diag_data = np.array(vectorizer_dict["idf_diag"])
        idf_diag_shape = (len(idf_diag_data), len(idf_diag_data))
        row = list(range(idf_diag_shape[0]))
        col = list(range(idf_diag_shape[0]))
        idf_diag = sp.csr_matrix((idf_diag_data, (row, col)),
                                 shape=idf_diag_shape)
        tfidf_transformer._idf_diag = idf_diag  # pylint: disable=W0212
    tfidf_vectorizer._tfidf = tfidf_transformer  # pylint: disable=W0212
    return tfidf_vectorizer
