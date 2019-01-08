from __future__ import division, unicode_literals

import json
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
from builtins import str
from future.utils import iteritems
from pathlib import Path
from scipy.sparse import dok_matrix, hstack
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.feature_extraction.text import (
    TfidfTransformer, TfidfVectorizer as SklearnTfidfVectorizer)
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_is_fitted
from snips_nlu_utils import normalize

from snips_nlu.constants import (
    DATA, END, ENTITIES, ENTITY, ENTITY_KIND,
    LANGUAGE, NGRAM, RES_MATCH_RANGE, RES_VALUE, START, TEXT)
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.entity_parser.builtin_entity_parser import (
    is_builtin_entity)
from snips_nlu.exceptions import NotTrained, _EmptyDatasetError
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.pipeline.configs.intent_classifier import (
    CooccurrenceVectorizerConfig, TfidfVectorizerConfig)
from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.preprocessing import stem, tokenize_light
from snips_nlu.resources import (
    get_stop_words, get_word_cluster)
from snips_nlu.slot_filler.features_utils import get_all_ngrams
from snips_nlu.utils import json_string, replace_entities_with_placeholders


class Featurizer(ProcessingUnit):
    """Feature extractor for text classification relying on ngrams tfidf and
    word cooccurrences features"""
    config_type = FeaturizerConfig
    unit_name = "featurizer"

    def __init__(self, config=None, **shared):
        # TODO: missing docstring
        if config is None:
            config = FeaturizerConfig()
        super(Featurizer, self).__init__(config, **shared)
        self.language = None
        self.tfidf_vectorizer = None
        self.cooccurrence_vectorizer = None
        self.builtin_entity_scope = None

    @property
    def fitted(self):
        if not self.tfidf_vectorizer or not self.tfidf_vectorizer.vocabulary:
            return False
        return True

    @property
    def feature_index_to_feature_name(self):
        """Maps the feature index of the feature matrix to printable features
         names, mainly useful for debug

            Returns:
                dict: a dict mapping feature indexes to printable features
                 names
        """
        if not self.fitted:
            return dict()

        index = {
            i: "ngram:%s" % ng
            for ng, i in iteritems(self.tfidf_vectorizer.vocabulary)
        }
        num_ng = len(index)
        if self.cooccurrence_vectorizer is not None:
            for word_pair, j in iteritems(
                    self.cooccurrence_vectorizer.word_pairs):
                index[j + num_ng] = "pair:%s+%s" % (word_pair[0], word_pair[1])
        return index

    def fit(self, dataset, utterances, classes, none_class):
        self.fit_transform(dataset, utterances, classes, none_class)
        return self

    def fit_transform(self, dataset, utterances, classes, none_class):
        self.language = dataset[LANGUAGE]

        utterances_texts = (get_text_from_chunks(u[DATA]) for u in utterances)
        if not any(tokenize_light(q, self.language) for q in utterances_texts):
            raise _EmptyDatasetError(
                "Couldn't fit because no utterance was found an")

        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        dataset_entities = set(dataset[ENTITIES])

        self.builtin_entity_scope = set(
            e for e in dataset_entities if is_builtin_entity(e))

        preprocessed = self._preprocess(utterances, training=True)
        preprocessed_utterances = preprocessed[0]
        preprocessed_utterances_texts = preprocessed[1]
        builtin_ents = preprocessed[2]
        custom_ents = preprocessed[3]
        w_clusters = preprocessed[4]

        tfidf_data = zip(
            preprocessed_utterances, builtin_ents, custom_ents, w_clusters)
        x_tfidf = self._fit_transform_tfidf_vectorizer(tfidf_data, classes)

        x = x_tfidf
        if self.config.added_cooccurrence_feature_ratio:
            cooccurrence_data = list(
                zip(preprocessed_utterances_texts, builtin_ents, custom_ents))
            # We fit the coocurrence matrix without noise
            non_null_cooccurrence_data, non_null_cooccurrence_classes = zip(*(
                (d, c) for d, c in zip(cooccurrence_data, classes)
                if c != none_class))
            self._fit_cooccurrence_vectorizer(
                non_null_cooccurrence_data, non_null_cooccurrence_classes)
            # We fit transform all the data
            x_cooccurrence = self.cooccurrence_vectorizer.transform(
                cooccurrence_data)

            x = hstack((x_tfidf, x_cooccurrence))

        return x

    def transform(self, utterances):
        preprocessed = self._preprocess(utterances)
        preprocessed_utterances = preprocessed[0]
        preprocessed_utterances_texts = preprocessed[1]
        builtin_ents = preprocessed[2]
        custom_ents = preprocessed[3]
        w_clusters = preprocessed[4]

        tfidf_data = zip(
            preprocessed_utterances, builtin_ents, custom_ents, w_clusters)
        x = self.tfidf_vectorizer.transform(tfidf_data)
        if self.cooccurrence_vectorizer:
            cooccurrence_data = zip(
                preprocessed_utterances_texts, builtin_ents, custom_ents)
            x_cooccurrence = self.cooccurrence_vectorizer.transform(
                cooccurrence_data)
            x = hstack((x, x_cooccurrence))
        return x

    def _fit_transform_tfidf_vectorizer(self, x, y):
        self.tfidf_vectorizer = TfidfVectorizer(
            self.config.tfidf_vectorizer_config)
        x_tfidf = self.tfidf_vectorizer.fit_transform(x, self.language)
        _, tfidf_pval = chi2(x_tfidf, y)
        best_tfidf_features = [i for i, v in enumerate(tfidf_pval)
                               if v < self.config.pvalue_threshold]
        if not best_tfidf_features:
            best_tfidf_features = [
                idx for idx, val in enumerate(tfidf_pval) if
                val == tfidf_pval.min()]
        best_tfidf_features = set(best_tfidf_features)
        best_ngrams = [ng for ng, i in
                       iteritems(self.tfidf_vectorizer.vocabulary)
                       if i in best_tfidf_features]
        self.tfidf_vectorizer.limit_vocabulary(best_ngrams)
        # We can't return x_tfidf[:best_tfidf_features] because of the
        # normalization in the tranform of the tfidf_vectorizer
        # this would lead to inconsistent result between: fit_transform(x, y)
        # and fit(x, y).transform(x)
        return self.tfidf_vectorizer.transform(x)

    def _fit_cooccurrence_vectorizer(self, x, y):
        self.cooccurrence_vectorizer = CooccurrenceVectorizer()
        x_cooccurrence = self.cooccurrence_vectorizer.fit_transform(
            x, self.language)
        _, pval = chi2(x_cooccurrence, y)
        top_k = int(self.config.added_cooccurrence_feature_ratio * len(
            self.tfidf_vectorizer.idf_diag))
        top_k_cooccurrence_ix = np.argpartition(
            pval, top_k, axis=None)[-top_k:]
        top_k_cooccurrence_ix = set(top_k_cooccurrence_ix)

        top_word_pairs = [
            pair for pair, i in iteritems(
                self.cooccurrence_vectorizer.word_pairs)
            if i in top_k_cooccurrence_ix
        ]

        self.cooccurrence_vectorizer.limit_word_pairs(top_word_pairs)
        return self

    def _preprocess(self, utterances, training=False):
        normalized_utterances = deepcopy(utterances)
        normalized_utterances_texts = []
        for u in normalized_utterances:
            for chunk in u[DATA]:
                chunk[TEXT] = _normalize_stem(chunk[TEXT], self.language,
                                              self.config.use_stemming)
            normalized_text = get_default_sep(self.language).join(
                chunk[TEXT] for chunk in u[DATA])
            normalized_utterances_texts.append(normalized_text)

        if training:
            builtin_ents, custom_ents = zip(*(
                _entities_from_utterance(u, self.config.use_stemming,
                                         self.language) for u in utterances))
        else:
            # Extract builtin entities
            builtin_ents = [
                self.builtin_entity_parser.parse(
                    u, self.builtin_entity_scope, use_cache=True)
                for u in normalized_utterances_texts
            ]

            custom_ents = [
                self.custom_entity_parser.parse(u, use_cache=True)
                for u in normalized_utterances_texts
            ]

        if self.config.word_clusters_name:
            original_utterances_text = [get_text_from_chunks(u[DATA])
                                        for u in utterances]
            w_clusters = [
                _get_word_cluster_features(
                    tokenize_light(u.lower(), self.language),
                    self.config.word_clusters_name,
                    self.language)
                for u in original_utterances_text
            ]
        else:
            w_clusters = [None for _ in normalized_utterances]

        return (normalized_utterances, normalized_utterances_texts,
                builtin_ents, custom_ents, w_clusters)

    def persist(self, path):
        path = Path(path)
        path.mkdir()

        # Persist the vectorizers
        tfidf_vectorizer = None
        if self.tfidf_vectorizer:
            tfidf_vectorizer = self.tfidf_vectorizer.unit_name
            tfidf_vectorizer_path = path / tfidf_vectorizer
            self.tfidf_vectorizer.persist(tfidf_vectorizer_path)

        cooccurrence_vectorizer = None
        if self.cooccurrence_vectorizer:
            cooccurrence_vectorizer = self.cooccurrence_vectorizer.unit_name
            cooccurrence_vectorizer_path = path / cooccurrence_vectorizer
            self.cooccurrence_vectorizer.persist(cooccurrence_vectorizer_path)

        builtin_scope = self.builtin_entity_scope
        if builtin_scope is not None:
            builtin_scope = list(builtin_scope)

        # Persist main object
        self_as_dict = {
            "language_code": self.language,
            "tfidf_vectorizer": tfidf_vectorizer,
            "cooccurrence_vectorizer": cooccurrence_vectorizer,
            "config": self.config.to_dict(),
            "builtin_entity_scope": builtin_scope
        }

        featurizer_path = type(self).build_unit_path(path)
        with featurizer_path.open("w", encoding="utf-8") as f:
            f.write(json_string(self_as_dict))

        # Persist metadata
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)

        featurizer_path = cls.build_unit_path(path)
        with featurizer_path.open("r", encoding="utf-8") as f:
            featurizer_dict = json.load(f)

        featurizer_config = featurizer_dict["config"]
        featurizer = cls(featurizer_config, **shared)

        featurizer.language = featurizer_dict["language_code"]

        tfidf_vectorizer = featurizer_dict["tfidf_vectorizer"]
        if tfidf_vectorizer:
            vectorizer_path = path / featurizer_dict["tfidf_vectorizer"]
            tfidf_vectorizer = TfidfVectorizer.from_path(
                vectorizer_path, **shared)
        featurizer.tfidf_vectorizer = tfidf_vectorizer

        cooccurrence_vectorizer = featurizer_dict["cooccurrence_vectorizer"]
        if cooccurrence_vectorizer:
            vectorizer_path = path / featurizer_dict["cooccurrence_vectorizer"]
            cooccurrence_vectorizer = CooccurrenceVectorizer.from_path(
                vectorizer_path, **shared)
        featurizer.cooccurrence_vectorizer = cooccurrence_vectorizer

        builtin_scope = featurizer_dict["builtin_entity_scope"]
        if builtin_scope is not None:
            builtin_scope = set(builtin_scope)
        featurizer.builtin_entity_scope = builtin_scope

        return featurizer


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


class TfidfVectorizer(ProcessingUnit):
    # TODO: missing doctring
    unit_name = "tfidf_vectorizer"
    config_type = TfidfVectorizerConfig

    def __init__(self, config=None, **shared):
        # TODO: missing doctring
        if config is None:
            config = TfidfVectorizerConfig()
        super(TfidfVectorizer, self).__init__(config, **shared)
        self._tfidf_vectorizer = None
        self._language = None

    def fit(self, x, language):
        self._language = language
        self._init_vectorizer()
        utterances = [
            self._enrich_utterance_for_tfidf(
                u, builtin_ents, custom_ents, w_clusters)
            for u, builtin_ents, custom_ents, w_clusters in x
        ]
        return self._tfidf_vectorizer.fit(utterances)

    def fit_transform(self, x, language):
        self._language = language
        self._init_vectorizer()
        utterances = [
            self._enrich_utterance_for_tfidf(
                u, builtin_ents, custom_ents, w_clusters)
            for u, builtin_ents, custom_ents, w_clusters in x
        ]
        return self._tfidf_vectorizer.fit_transform(utterances)

    def fitted(self):
        return self._tfidf_vectorizer is not None and hasattr(
            self._tfidf_vectorizer, "vocabulary_")

    def transform(self, x):
        utterances = [self._enrich_utterance_for_tfidf(*data) for data in x]
        return self._tfidf_vectorizer.transform(utterances)

    def _enrich_utterance_for_tfidf(self, utterance, builtin_entities,
                                    custom_entities, word_clusters):
        custom_entities_features = [
            _entity_name_to_feature(e[ENTITY_KIND], self.language)
            for e in custom_entities]

        builtin_entities_features = [
            _builtin_entity_to_feature(ent[ENTITY_KIND], self.language)
            for ent in builtin_entities
        ]

        # We remove values of builtin slots from the utterance to avoid
        # learning specific samples such as '42' or 'tomorrow'
        filtered_normalized_stemmed_tokens = [
            chunk[TEXT] for chunk in utterance[DATA]
            if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
        ]

        features = get_default_sep(self.language).join(
            filtered_normalized_stemmed_tokens)

        if builtin_entities_features:
            features += " " + " ".join(sorted(builtin_entities_features))
        if custom_entities_features:
            features += " " + " ".join(sorted(custom_entities_features))
        if word_clusters:
            features += " " + " ".join(sorted(word_clusters))

        return features

    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere
        # than in the fit
        return self._language

    @property
    def vocabulary(self):
        if self._tfidf_vectorizer and hasattr(
                self._tfidf_vectorizer, "vocabulary_"):
            return self._tfidf_vectorizer.vocabulary_
        return None

    # pylint: disable=protected-access
    def limit_vocabulary(self, ngrams):
        """Set the vectorizer vocabulary by limiting it to the given ngrams
            Args:
             ngrams (iterable of str or tuples of str): ngrams to keep

            Returns:
                .TfidfVectorizer: the vectorizer with limited vocabulary
        """
        try:
            check_is_fitted(self._tfidf_vectorizer, "vocabulary_")
        except (SklearnNotFittedError, AttributeError):
            raise NotTrained(
                "vectorizer must be fitted before limiting its vocabulary")

        ngrams = set(ngrams)
        existing_ngrams = set(self.vocabulary)
        extra_values = ngrams - existing_ngrams

        if extra_values:
            raise ValueError("Invalid ngrams %s, expected values in %s"
                             % (sorted(extra_values), sorted(existing_ngrams)))

        new_ngrams, new_index = zip(*sorted(
            (ng, self.vocabulary[ng]) for ng in ngrams))
        if not new_ngrams:
            raise ValueError(
                "No feature remain after limiting the vocabulary, please use "
                "different ngram"
            )

        self._tfidf_vectorizer.vocabulary_ = {
            ng: new_i for new_i, ng in enumerate(new_ngrams)
        }
        # The new_idf_data is valid because the previous _idf_diag was indexed
        # with sorted ngrams and new_index is also indexed with sorted ngrams
        new_idf_data = self._tfidf_vectorizer._tfidf._idf_diag.data[
            list(new_index)]
        self._tfidf_vectorizer._tfidf._idf_diag = sp.spdiags(
            new_idf_data, diags=0, m=len(new_index), n=len(new_index),
            format="csr")
        return self

    @property
    def idf_diag(self):
        if self._tfidf_vectorizer and hasattr(
                self._tfidf_vectorizer, "vocabulary_"):
            return self._tfidf_vectorizer.idf_
        return None

    def _init_vectorizer(self):
        if not self._language:
            raise NotTrained("_language has not been set")
        self._tfidf_vectorizer = SklearnTfidfVectorizer(
            tokenizer=lambda x: tokenize_light(x, self._language))
        return self

    def persist(self, path):
        path = Path(path)

        _vectorizer = None
        if self._tfidf_vectorizer is not None:
            vocab = {k: int(v) for k, v in iteritems(self.vocabulary)}
            idf_diag = self.idf_diag.tolist()
            _vectorizer = {
                "vocab": vocab,
                "idf_diag": idf_diag
            }

        self_as_dict = {
            "vectorizer": _vectorizer,
            "language_code": self.language,
            "config": self.config.to_dict(),
        }

        path.mkdir()
        vectorizer_path = type(self).build_unit_path(path)
        with vectorizer_path.open("w", encoding="utf-8") as f:
            f.write(json_string(self_as_dict))
        self.persist_metadata(path)

    @classmethod
    # pylint: disable=W0212
    def from_path(cls, path, **shared):
        path = Path(path)
        vectorizer_path = cls.build_unit_path(path)

        with vectorizer_path.open("r", encoding="utf-8") as f:
            vectorizer_dict = json.load(f)

        vectorizer = cls(vectorizer_dict["config"], **shared)
        vectorizer._language = vectorizer_dict["language_code"]

        _vectorizer = vectorizer_dict["vectorizer"]
        if _vectorizer:
            vocab = _vectorizer["vocab"]
            idf_diag_data = _vectorizer["idf_diag"]
            idf_diag_data = np.array(idf_diag_data)

            idf_diag_shape = (len(idf_diag_data), len(idf_diag_data))
            row = list(range(idf_diag_shape[0]))
            col = list(range(idf_diag_shape[0]))
            idf_diag = sp.csr_matrix(
                (idf_diag_data, (row, col)), shape=idf_diag_shape)

            tfidf_transformer = TfidfTransformer()
            tfidf_transformer._idf_diag = idf_diag

            _vectorizer = SklearnTfidfVectorizer(
                tokenizer=lambda x: tokenize_light(x, vectorizer._language))
            _vectorizer.vocabulary_ = vocab

            _vectorizer._tfidf = tfidf_transformer

        vectorizer._tfidf_vectorizer = _vectorizer
        return vectorizer


class CooccurrenceVectorizer(ProcessingUnit):
    unit_name = "cooccurrence_vectorizer"
    config_type = CooccurrenceVectorizerConfig

    def __init__(self, config=None, **shared):
        if config is None:
            config = CooccurrenceVectorizerConfig()
        super(CooccurrenceVectorizer, self).__init__(config, **shared)
        self._word_pairs = None
        self._language = None

    def fit(self, x, language):
        """Fits the CooccurrenceVectorizer

        Given list of utterances the CooccurrenceVectorizer will extract word
        pairs appearing in the same utterance. The order in which the words
        appear is kept. Additionally, if self.config.window_size is not None
        then the vectorizer will only look in a context window of
        self.config.window_size after each word.

            Args:
             x (iterable): iterable of 3-tuples of the form
              (tokenized_utterances, builtin_entities, custom_entities)
             language (str): language used in the utterances
            Returns:
             self: the fitted CooccurrenceVectorizer
        """
        self._language = language
        utterances = self._preprocess(list(x))
        word_pairs = set()
        for u in utterances:
            for i, w1 in enumerate(u):
                max_index = None
                if self.config.window_size is not None:
                    max_index = i + self.config.window_size + 1
                for w2 in u[i + 1:max_index]:
                    word_pairs.add((w1, w2))
        self._word_pairs = {
            pair: i for i, pair in enumerate(sorted(word_pairs))
        }
        return self

    def fitted(self):
        return self.word_pairs is not None

    def fit_transform(self, x, language):
        """Fits the vectorizer and returns the feature matrix"""
        return self.fit(x, language).transform(x)

    def _preprocess(self, x):
        if not x:
            return []
        utterances, builtin_ents, custom_ents = zip(*x)

        all_entities = (b + c for b, c in zip(builtin_ents, custom_ents))
        placeholder_fn = self._get_placeholder_fn()
        tokenized_utterances = [
            replace_entities_with_placeholders(u, ents, placeholder_fn)[1]
            for u, ents in zip(utterances, all_entities)
        ]

        tokenized_utterances = [tokenize_light(u, self.language)
                                for u in tokenized_utterances]

        if self.config.unknown_words_replacement_string:
            tokenized_utterances = [
                [t for t in u if
                 t != self.config.unknown_words_replacement_string]
                for u in tokenized_utterances
            ]

        return tokenized_utterances

    def transform(self, x):
        """Computes the cooccurrence feature matrix.
            Args:
             x (iterable): iterable of 3-tuples of the form
              (tokenized_utterances, builtin_entities, custom_entities)
            Returns:
             A sparse matrix X of shape (len(x), len(self.word_pairs)) where
             X[i, j] = 1.0 if x[i][0] contains the contains the words
             cooccurrence (w1, w2) and if self.word_paris[(w1, w2)] = j
        """
        if self.word_pairs is None:
            raise NotTrained("CooccurrenceVectorizer must be fitted")
        x = list(x)
        utterances = self._preprocess(x)
        x_coo = dok_matrix((len(x), len(self.word_pairs)), dtype=np.int32)
        if self.config.filter_stop_words:
            stop_words = get_stop_words(self.language)
            utterances = ([t for t in u if t not in stop_words]
                          for u in utterances)
        for i, u in enumerate(utterances):
            for j, w1 in enumerate(u):
                max_index = None
                if self.config.window_size is not None:
                    max_index = j + self.config.window_size + 1
                for w2 in u[j + 1:max_index]:
                    if (w1, w2) in self.word_pairs:
                        x_coo[i, self.word_pairs[(w1, w2)]] = 1

        return x_coo.tocsr()

    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere
        # than in the fit
        return self._language

    @property
    def word_pairs(self):
        return self._word_pairs

    def limit_word_pairs(self, word_pairs):
        """
        Set the vectorizer word_pairs to be equal to the by limiting it to the
         given word_pairs
            Args:
                word_pairs (iterable of 2-tuples (str, str)): word_pairs to
                 keep

            Returns:
                .CooccurrenceVectorizer: the vectorizer with limited word pairs
        """

        if self._word_pairs is None:
            raise NotTrained(
                "vectorizer must be fitted before limiting its words pairs")

        word_pairs = set(word_pairs)
        existing_pairs = set(self.word_pairs)
        extra_values = word_pairs - existing_pairs

        if extra_values:
            raise ValueError(
                "Invalid word pairs %s, expected values in %s"
                % (sorted(extra_values), sorted(existing_pairs))
            )

        self._word_pairs = {
            ng: new_i for new_i, ng in enumerate(sorted(word_pairs))
        }

        if not self.word_pairs:
            raise ValueError(
                "No feature remain after limiting the vocabulary, please use "
                "different word_pairs"
            )
        return self

    def _get_placeholder_fn(self):
        if not self.language:
            raise NotTrained("Vectorizer must be fitted before getting the "
                             "placeholder function")

        def fn(entity_name):
            return "".join(
                tokenize_light(str(entity_name), str(self.language))).upper()

        return fn

    def persist(self, path):
        path = Path(path)
        path.mkdir()

        self_as_dict = {
            "language_code": self.language,
            "word_pairs": {
                i: list(p) for p, i in iteritems(self.word_pairs)
            },
            "config": self.config.to_dict()
        }
        vectorizer_json = json_string(self_as_dict)
        vectorizer_path = type(self).build_unit_path(path)
        with vectorizer_path.open(mode="w") as f:
            f.write(vectorizer_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)
        vectorizer_path = cls.build_unit_path(path)
        if not vectorizer_path.exists():
            raise OSError("Missing vectorizer file: %s" % vectorizer_path.name)

        with vectorizer_path.open(encoding="utf8") as f:
            vectorizer_dict = json.load(f)
        config = vectorizer_dict.pop("config")

        self = cls(config, **shared)
        self._language = vectorizer_dict["language_code"]
        self._word_pairs = None
        if vectorizer_dict["word_pairs"]:
            self._word_pairs = {
                tuple(p): int(i)
                for i, p in iteritems(vectorizer_dict["word_pairs"])
            }
        return self


def _entities_from_utterance(utterance, use_stemming, language):
    builtin_ents = []
    custom_ents = []
    current_ix = 0
    for i, chunk in enumerate(utterance[DATA]):
        text = _normalize_stem(chunk[TEXT], language, use_stemming)
        if i != 0:
            current_ix += 1  # normalized stemmed tokens are joined with a " "
        text_length = len(text)
        if ENTITY in chunk:
            ent = {
                ENTITY_KIND: chunk[ENTITY],
                RES_VALUE: text,
                RES_MATCH_RANGE: {
                    START: current_ix,
                    END: current_ix + text_length
                }
            }
            if is_builtin_entity(ent[ENTITY_KIND]):
                builtin_ents.append(ent)
            else:
                custom_ents.append(ent)
        current_ix += text_length
    return builtin_ents, custom_ents
