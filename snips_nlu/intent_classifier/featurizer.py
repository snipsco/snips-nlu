from __future__ import division, unicode_literals

import numpy as np
import scipy.sparse as sp
from builtins import object, range
from future.utils import iteritems
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_is_fitted
from snips_nlu_utils import normalize

from snips_nlu.constants import (BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER,
                                 CUSTOM_ENTITY_PARSER_USAGE, DATA, END,
                                 ENTITY_KIND, NGRAM, RES_MATCH_RANGE, START,
                                 VALUE, TEXT, ENTITY, ENTITIES)
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.entity_parser.builtin_entity_parser import (BuiltinEntityParser,
                                                           is_builtin_entity)
from snips_nlu.entity_parser.custom_entity_parser import CustomEntityParser
from snips_nlu.intent_parser.deterministic_intent_parser import \
    _deduplicate_overlapping_entities
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.preprocessing import stem, tokenize_light
from snips_nlu.resources import (
    get_stop_words, get_word_cluster)
from snips_nlu.slot_filler.features_utils import get_all_ngrams


class Featurizer(object):
    def __init__(self, language, unknown_words_replacement_string,
                 config=FeaturizerConfig(), tfidf_vectorizer=None,
                 best_features=None,
                 builtin_entity_parser=None, custom_entity_parser=None):
        self.config = config
        self.language = language
        if tfidf_vectorizer is None:
            tfidf_vectorizer = _get_tfidf_vectorizer(
                self.language, sublinear_tf=self.config.sublinear_tf)

        self.kmeans_tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: tokenize_light(x, language),
            ngram_range=(1, 3),
            sublinear_tf=self.config.sublinear_tf)

        self.tfidf_vectorizer = tfidf_vectorizer
        self.best_features = best_features
        self.unknown_words_replacement_string = \
            unknown_words_replacement_string

        self.builtin_entity_parser = builtin_entity_parser
        self.custom_entity_parser = custom_entity_parser

    @property
    def fitted(self):
        try:
            check_is_fitted(self.tfidf_vectorizer, 'vocabulary_')
            return True
        except NotFittedError:
            return False

    def fit(self, dataset, utterances, classes):
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        utterances_texts = (get_text_from_chunks(u[DATA]) for u in utterances)
        if not any(tokenize_light(q, self.language) for q in utterances_texts):
            return None

        preprocessed_utterances = self.preprocess_utterances(utterances)

        X_clusterer = self.kmeans_tfidf_vectorizer.fit_transform(
            preprocessed_utterances)

        # We hope to find 5 formulation per intents
        target_num_clusters = len(dataset["intents"]) * 5
        if len(preprocessed_utterances) > target_num_clusters:
            n_clusters = target_num_clusters
        else:
            n_clusters = len(dataset["intents"])

        self.kmeans_clusterer = KMeans(n_clusters=n_clusters, n_jobs=-1)
        X_train_clusters = self.kmeans_clusterer.fit_transform(X_clusterer)

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

        self.entities = {e: i for i, e in enumerate(dataset[ENTITIES])}
        num_entities = len(self.entities)
        min_cluster_ix = X_train_tfidf.shape[1]
        max_cluster_ix = min_cluster_ix + X_train_clusters.shape[1] + num_entities
        for i in range(X_train_tfidf.shape[0], max_cluster_ix):
            self.best_features.append(i)
        return self

    def transform(self, utterances):
        preprocessed_utterances = self.preprocess_utterances(utterances)

        X_clusterer = self.kmeans_tfidf_vectorizer.transform(
            preprocessed_utterances)
        X_clusters = self.kmeans_clusterer.transform(X_clusterer)

        # pylint: disable=C0103
        X_tfidf = self.tfidf_vectorizer.transform(
            preprocessed_utterances)

        X_entities = np.zeros((len(utterances), len(self.entities)),
                              dtype=np.float)
        for i, u in enumerate(utterances):
            utterance_text = get_text_from_chunks(u[DATA])
            utterance_tokens = tokenize_light(utterance_text, self.language)
            normalized_stemmed_tokens = [
                _normalize_stem(t, self.language,  self.config.use_stemming)
                for t in utterance_tokens]
            custom_entities = self.custom_entity_parser.parse(
                " ".join(normalized_stemmed_tokens))

            builtin_entities = self.custom_entity_parser.parse(
                " ".join(normalized_stemmed_tokens))
            entities = custom_entities + builtin_entities
            for ent in entities:
                ent_ix = self.entities[ent[ENTITY_KIND]]
                X_entities[i, ent_ix] += 1.0

        X_train = hstack(
            (X_tfidf, csr_matrix(X_clusters), csr_matrix(X_entities)),
            format="csr")

        X = X_train[:, self.best_features]
        # pylint: enable=C0103
        return X

    def fit_transform(self, dataset, queries, y):
        return self.fit(dataset, queries, y).transform(queries)

    def preprocess_utterances(self, utterances):
        return [
            _preprocess_utterance(
                u, self.language, self.config.word_clusters_name,
                self.config.use_stemming
            )
            for u in utterances
        ]

    def fit_builtin_entity_parser_if_needed(self, dataset):
        # We only fit a builtin entity parser when the unit has already been
        # fitted or if the parser is none.
        # In the other cases the parser is provided fitted by another unit.
        if self.builtin_entity_parser is None or self.fitted:
            self.builtin_entity_parser = BuiltinEntityParser.build(
                dataset=dataset)
        return self

    def fit_custom_entity_parser_if_needed(self, dataset):
        # We only fit a custom entity parser when the unit has already been
        # fitted or if the parser is none.
        # In the other cases the parser is provided fitted by another unit.
        required_resources = self.config.get_required_resources()
        if not required_resources:
            return self
        parser_usage = required_resources.get(CUSTOM_ENTITY_PARSER_USAGE)
        if not parser_usage:
            return self

        if self.custom_entity_parser is None or self.fitted:
            self.custom_entity_parser = CustomEntityParser.build(
                dataset, parser_usage)
        return self

    def to_dict(self):
        """Returns a json-serializable dict"""
        if hasattr(self.tfidf_vectorizer, "vocabulary_"):
            # pylint: # pylint: disable=W0212
            vocab = {k: int(v) for k, v in
                     iteritems(self.tfidf_vectorizer.vocabulary_)}
            idf_diag = self.tfidf_vectorizer._tfidf._idf_diag.data.tolist()
        else:
            vocab = None
            idf_diag = None

        tfidf_vectorizer = {
            "vocab": vocab,
            "idf_diag": idf_diag
        }

        return {
            "language_code": self.language,
            "tfidf_vectorizer": tfidf_vectorizer,
            "best_features": self.best_features,
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
        self = cls(
            language=language,
            tfidf_vectorizer=tfidf_vectorizer,
            best_features=obj_dict["best_features"],
            config=config,
            unknown_words_replacement_string=obj_dict[
                "unknown_words_replacement_string"],
            builtin_entity_parser=shared.get(BUILTIN_ENTITY_PARSER),
            custom_entity_parser=shared.get(CUSTOM_ENTITY_PARSER)
        )
        return self


def _preprocess_utterance(utterance, language, word_clusters_name,
                          use_stemming):
    utterance_text = get_text_from_chunks(utterance[DATA])
    utterance_tokens = tokenize_light(utterance_text, language)
    word_clusters_features = _get_word_cluster_features(
        utterance_tokens, word_clusters_name, language)
    # normalized_stemmed_tokens = [_normalize_stem(t, language, use_stemming)
    #                              for t in utterance_tokens]
    #
    # custom_entities = custom_entity_parser.parse(
    #     " ".join(normalized_stemmed_tokens))
    # custom_entities = [e for e in custom_entities
    #                    if e["value"] != unknownword_replacement_string]
    # custom_entities_features = [
    #     _entity_name_to_feature(e[ENTITY_KIND], language)
    #     for e in custom_entities]
    #
    # builtin_entities = builtin_entity_parser.parse(
    #     utterance_text, use_cache=True)
    # builtin_entities_features = [
    #     _builtin_entity_to_feature(ent[ENTITY_KIND], language)
    #     for ent in builtin_entities
    # ]

    # We remove values of builtin slots from the utterance to avoid learning
    # specific samples such as '42' or 'tomorrow'
    filtered_normalized_stemmed_tokens = [
        _normalize_stem(chunk[TEXT], language, use_stemming)
        for chunk in utterance[DATA]
        if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
    ]

    features = get_default_sep(language).join(
        filtered_normalized_stemmed_tokens)
    # if builtin_entities_features:
    #     features += " " + " ".join(sorted(builtin_entities_features))
    # if custom_entities_features:
    #     features += " " + " ".join(sorted(custom_entities_features))
    if word_clusters_features:
        features += " " + " ".join(sorted(word_clusters_features))

    return features


def _add_entities_features(query, entities, unknownword_replacement_string,
                           language):
    entities = [e for e in entities
                if e["value"] != unknownword_replacement_string]
    features = "BOSSYMBOL "
    cur_ix = 0
    for ent in entities:
        start, end = (ent[RES_MATCH_RANGE][START], ent[RES_MATCH_RANGE][END])
        if start > cur_ix:
            features += query[cur_ix:start]
        if is_builtin_entity(ent[ENTITY_KIND]):
            placeholder = _builtin_entity_to_feature(
                ent[ENTITY_KIND], language)
        else:
            placeholder = _entity_name_to_feature(ent[ENTITY_KIND], language)
        features += placeholder
        cur_ix = end

    if cur_ix < len(query):
        features += query[cur_ix:]

    features += " EOSSYMBOL"

    # Add all the custom matched values at the end of the query
    for ent in entities:
        if not is_builtin_entity(ent[ENTITY_KIND]):
            features += " %s" % ent[VALUE]
    return features


def _get_tfidf_vectorizer(language, sublinear_tf=False):
    return TfidfVectorizer(tokenizer=lambda x: tokenize_light(x, language),
                           sublinear_tf=sublinear_tf)


def _get_tokens_clusters(tokens, language, cluster_name):
    clusters = get_word_cluster(language, cluster_name)
    return [clusters[t] for t in tokens if t in clusters]


def _entity_name_to_feature(entity_name, language):
    return "entityfeature%s" % "".join(tokenize_light(
        entity_name.lower(), language))


def _builtin_entity_to_feature(builtin_entity_label, language):
    return "builtinentityfeature%s" % "".join(tokenize_light(
        builtin_entity_label.lower(), language))


def _normalize_stem(text, language, use_stemming):
    if use_stemming:
        return stem(text, language)
    return normalize(text)


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
