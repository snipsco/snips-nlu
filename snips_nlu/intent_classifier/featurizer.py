from __future__ import division, unicode_literals

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack, dok_matrix

from future.utils import iteritems, itervalues
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_is_fitted
from snips_nlu_utils import normalize

from snips_nlu.constants import (
    BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER, CUSTOM_ENTITY_PARSER_USAGE,
    DATA, ENTITY, ENTITY_KIND, NGRAM, TEXT, LANGUAGE, ENTITIES, RES_VALUE,
    RES_MATCH_RANGE, START, END)
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.entity_parser.builtin_entity_parser import (BuiltinEntityParser,
                                                           is_builtin_entity)
from snips_nlu.entity_parser.custom_entity_parser import CustomEntityParser
from snips_nlu.exceptions import _EmptyDataError, NotTrained
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.pipeline.configs.intent_classifier import \
    CooccurrenceVectorizerConfig
from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.preprocessing import stem, tokenize_light
from snips_nlu.resources import (
    get_stop_words, get_word_cluster)
from snips_nlu.slot_filler.features_utils import get_all_ngrams
from snips_nlu.utils import replace_entities_with_placeholders, json_string


class Featurizer(object):
    def __init__(self, language=None, config=FeaturizerConfig(),
                 tfidf_vectorizer=None,
                 unknown_words_replacement_string=None,
                 cooccurrence_vectorizer=None,
                 builtin_entity_parser=None, custom_entity_parser=None,
                 builtin_entity_scope=None):
        self.config = config
        self.language = language
        self.tfidf_vectorizer = tfidf_vectorizer
        self.language = language
        self.cooccurrence_vectorizer = cooccurrence_vectorizer
        self.unknown_words_replacement_string = \
            unknown_words_replacement_string

        self.builtin_entity_parser = builtin_entity_parser
        self.custom_entity_parser = custom_entity_parser

        self.builtin_entity_scope = builtin_entity_scope

    @property
    def fitted(self):
        if not self.tfidf_vectorizer:
            return False
        try:
            check_is_fitted(self.tfidf_vectorizer, 'vocabulary_')
            return True
        except SklearnNotFittedError:
            return False

    def fit(self, dataset, utterances, classes):
        self.fit_transform(dataset, utterances, classes)
        return self

    def fit_transform(self, dataset, utterances, classes):
        self.language = dataset[LANGUAGE]
        self.tfidf_vectorizer = _get_tfidf_vectorizer(
            self.language, sublinear_tf=self.config.sublinear_tf)

        utterances_texts = (get_text_from_chunks(u[DATA]) for u in utterances)
        if not any(tokenize_light(q, self.language) for q in utterances_texts):
            raise _EmptyDataError(
                "Couldn't fit because no utterance was found an")

        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        dataset_entities = set(dataset[ENTITIES])

        self.builtin_entity_scope = set(
            e for e in dataset_entities if is_builtin_entity(e))

        self.none_class_ix = max(classes)

        normalized, builtin_ents, custom_ents, w_clusters = self. \
            _preprocess(utterances, training=True)
        tfidf_data = zip(utterances, builtin_ents, custom_ents, w_clusters)
        x_tfidf = self._fit_transform_tfidf_vectorizer(tfidf_data, classes)

        x = x_tfidf
        if self.config.added_cooccurrence_feature_ratio:
            cooccurrence_data = list(
                zip(normalized, builtin_ents, custom_ents))
            # We fit the coocurrence matrix without noise
            non_null_cooccurrence_data, non_null_cooccurrence_classes = zip(*(
                (d, c) for d, c in zip(cooccurrence_data, classes)
                if c != self.none_class_ix))
            self._fit_cooccurrence_vectorizer(
                non_null_cooccurrence_data, non_null_cooccurrence_classes)
            # We fit transform all the data
            x_cooccurrence = self.cooccurrence_vectorizer.transform(
                cooccurrence_data)

            x = hstack((x_tfidf, x_cooccurrence))

        return x

    def transform(self, utterances):
        normalized_utterances, builtin_ents, custom_ents, w_clusters = self. \
            _preprocess(utterances)

        tfidf_data = zip(utterances, builtin_ents, custom_ents, w_clusters)

        enriched_utterances = (self._enrich_utterance_for_tfidf(*d)
                               for d in tfidf_data)

        x = self.tfidf_vectorizer.transform(enriched_utterances)
        if self.cooccurrence_vectorizer:
            cooccurrence_data = zip(normalized_utterances, builtin_ents,
                                    custom_ents)
            x_cooccurrence = self.cooccurrence_vectorizer.transform(
                cooccurrence_data)
            x = hstack((x, x_cooccurrence))
        return x

    def _preprocess(self, utterances, training=False):
        text_utterances = [get_text_from_chunks(u[DATA]) for u in utterances]

        normalized_utterances = [
            _normalize_stem(u, self.language, self.config.use_stemming)
            for u in text_utterances
        ]

        if training:
            builtin_ents, custom_ents = zip(*(
                _entities_from_utterance(u, self.config.use_stemming,
                                         self.language) for u in utterances))
        else:
            # Extract builtin entities
            builtin_ents = [
                self.builtin_entity_parser.parse(
                    u, self.builtin_entity_scope, use_cache=True)
                for u in normalized_utterances
            ]

            custom_ents = [
                self.custom_entity_parser.parse(u, use_cache=True)
                for u in normalized_utterances
            ]

        if self.config.word_clusters_name:
            w_clusters = [
                _get_word_cluster_features(
                    tokenize_light(u.lower(), self.language),
                    self.config.word_clusters_name,
                    self.language)
                for u in text_utterances
            ]
        else:
            w_clusters = [None for _ in text_utterances]

        return (normalized_utterances, builtin_ents, custom_ents, w_clusters)

    def _fit_transform_tfidf_vectorizer(self, x, y):
        featurized_utterances = [
            self._enrich_utterance_for_tfidf(
                utterance, builtin_ents, customs_ents, w_clusters)
            for utterance, builtin_ents, customs_ents, w_clusters in x
        ]

        x_tfidf = self.tfidf_vectorizer.fit_transform(featurized_utterances, y)
        _, tfidf_pval = chi2(x_tfidf, y)
        best_tfidf_features = [i for i, v in enumerate(tfidf_pval)
                               if v < self.config.pvalue_threshold]
        if not best_tfidf_features:
            best_tfidf_features = [
                idx for idx, val in enumerate(tfidf_pval) if
                val == tfidf_pval.min()]
        self._limit_tfidf_vectorizer_vocabulary(best_tfidf_features)
        # We can't return x_tfidf[:best_tfidf_features] because of the
        # normalization in the tranform of the tfidf_vectorizer
        # this would lead to inconsistent result between: fit_transform(x, y)
        # and fit(x, y).transform(x)
        return self.tfidf_vectorizer.transform(featurized_utterances)

    # pylint: disable=protected-access
    def _limit_tfidf_vectorizer_vocabulary(self, ngram_indexes):
        """Set the vectorizer vocabulary by limiting it to the n-grams
        which index is contained in ngram_indexes
            Args:
             ngram_indexes (iterable of int): indexes of the ngrams to keep
            Returns:
                self: the vectorizer with a limited vocabulary
        :return:
        """
        try:
            check_is_fitted(self.tfidf_vectorizer, "vocabulary_")
        except SklearnNotFittedError:
            raise NotTrained(
                "vectorizer must be fitted before limiting its vocabulary")

        ngram_indexes = set(ngram_indexes)
        existing_ix = set(itervalues(self.tfidf_vectorizer.vocabulary_))

        extra_values = ngram_indexes - existing_ix

        # Restrict the vectorizer vocabulary and remove corresponding indexes
        # from the transformer's idf diag

        if extra_values:
            raise ValueError("Invalid ngrams indexes %s, expected values in"
                             " %s" % (extra_values, existing_ix))

        new_ngrams, new_index = zip(*sorted(
            (ng, i) for ng, i in iteritems(self.tfidf_vectorizer.vocabulary_)
            if i in ngram_indexes))

        if not new_ngrams:
            raise ValueError(
                "No feature remain after limiting the vocabulary, please use "
                "different ngram_indexes"
            )

        self.tfidf_vectorizer.vocabulary_ = {
            ng: new_i for new_i, ng in enumerate(new_ngrams)
        }
        new_idf_data = self.tfidf_vectorizer._tfidf._idf_diag.data[
            list(new_index)]
        self.tfidf_vectorizer._tfidf._idf_diag = sp.spdiags(
            new_idf_data, diags=0, m=len(new_index), n=len(new_index),
            format="csr")
        return self

    def _fit_cooccurrence_vectorizer(self, x, y):
        self.cooccurrence_vectorizer = CooccurrenceVectorizer()
        x_cooccurrence = self.cooccurrence_vectorizer.fit_transform(
            x, self.language)
        _, pval = chi2(x_cooccurrence, y)
        top_k = int(self.config.added_cooccurrence_feature_ratio * len(
            self.tfidf_vectorizer.idf_))
        top_k_cooccurrence_ix = np.argpartition(pval, top_k, axis=None)
        self.cooccurrence_vectorizer.limit_vocabulary(top_k_cooccurrence_ix)
        return self

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
            _normalize_stem(
                chunk[TEXT], self.language, self.config.use_stemming)
            for chunk in utterance[DATA]
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

    def to_dict(self):
        """Returns a json-serializable dict"""
        if hasattr(self.tfidf_vectorizer, "vocabulary_"):
            # pylint: # pylint: disable=W0212
            vocab = {k: int(v) for k, v in
                     iteritems(self.tfidf_vectorizer.vocabulary_)}
            idf_diag = self.tfidf_vectorizer.idf_.tolist()
        else:
            vocab = None
            idf_diag = None

        tfidf_vectorizer = {
            "vocab": vocab,
            "idf_diag": idf_diag
        }

        builtin_scope = self.builtin_entity_scope
        if builtin_scope is not None:
            builtin_scope = list(builtin_scope)

        return {
            "language_code": self.language,
            "tfidf_vectorizer": tfidf_vectorizer,
            "config": self.config.to_dict(),
            "unknown_words_replacement_string":
                self.unknown_words_replacement_string,
            "builtin_entity_scope": builtin_scope
        }

    @classmethod
    def from_dict(cls, obj_dict, **shared):
        """Creates a :class:`Featurizer` instance from a :obj:`dict`

        The dict must have been generated with :func:`~Featurizer.to_dict`
        """
        language = obj_dict.pop("language_code")
        config = FeaturizerConfig.from_dict(obj_dict.pop("config"))
        tfidf_vectorizer = _deserialize_tfidf_vectorizer(
            obj_dict.pop("tfidf_vectorizer"), language, config.sublinear_tf)

        builtin_scope = obj_dict.pop("builtin_entity_scope")
        if builtin_scope is not None:
            builtin_scope = set(builtin_scope)

        self = cls(
            language=language,
            tfidf_vectorizer=tfidf_vectorizer,
            config=config,
            builtin_entity_parser=shared.get(BUILTIN_ENTITY_PARSER),
            custom_entity_parser=shared.get(CUSTOM_ENTITY_PARSER),
            builtin_entity_scope=builtin_scope,
            **obj_dict
        )
        return self


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


class CooccurrenceVectorizer(ProcessingUnit):
    unit_name = "cooccurrence_vectorizer"
    config_type = CooccurrenceVectorizerConfig

    def __init__(self, config=None, **shared):
        if config is None:
            config = CooccurrenceVectorizerConfig()
        super(CooccurrenceVectorizer, self).__init__(config, **shared)
        self.word_pairs = None
        self.language = None

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
        self.language = language
        utterances = self._preprocess(list(x))

        self.word_pairs = dict()
        word_pairs = set()
        for u in utterances:
            for i, w1 in enumerate(u):
                max_index = None
                if self.config.window_size is not None:
                    max_index = i + self.config.window_size + 1
                for w2 in u[i + 1:max_index]:
                    word_pairs.add((w1, w2))
        self.word_pairs = {
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

        placeholder_fn = lambda entity_name: "".join(
            tokenize_light(str(entity_name), str(self.language))).upper()

        all_entities = (b + c for b, c in zip(builtin_ents, custom_ents))
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
        if self.config.use_stop_words:
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

    def limit_vocabulary(self, word_pair_indexes):
        """Set the vectorizer vocabulary by limiting it to the words pairs
        which index is contained in word_pair_indexes
            Args:
             word_pair_indexes (iterable of int): indexes of the word_pairs to
              keep
            Returns:
                self: the vectorizer with a limited vocabulary
        :return:
        """

        word_pair_indexes = set(word_pair_indexes)
        existing_ix = set(itervalues(self.word_pairs))

        extra_values = word_pair_indexes - existing_ix

        if extra_values:
            raise ValueError("Invalid word pairs indexes %s, expected values "
                             "in %s" % (extra_values, existing_ix))

        new_pairs = (p for p, i in iteritems(self.word_pairs)
                     if i in word_pair_indexes)
        self.word_pairs = {
            pair: i for i, pair in enumerate(sorted(new_pairs))
        }
        if not self.word_pairs:
            raise ValueError(
                "No feature remain after limiting the vocabulary, please use "
                "different word_pair_indexes"
            )
        return self

    def persist(self, path):
        path = Path(path)
        path.mkdir()

        self_as_dict = {
            "language": self.language,
            "word_pairs": {
                i: list(p) for p, i in iteritems(self.word_pairs)
            },
            "config": self.config.to_dict()
        }
        vectorizer_json = json_string(self_as_dict)
        with (path / ("%s.json" % self.unit_name)).open(mode="w") as f:
            f.write(vectorizer_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)
        vectorizer_path = path / ("%s.json" % cls.unit_name)
        if not vectorizer_path.exists():
            raise OSError("Missing vectorizer file: %s" % vectorizer_path.name)

        with vectorizer_path.open(encoding="utf8") as f:
            vectorizer_dict = json.load(f)
        config = vectorizer_dict.pop("config")

        self = cls(config, **shared)
        self.language = vectorizer_dict["language"]
        self.word_pairs = None
        if vectorizer_dict["word_pairs"]:
            self.word_pairs = {
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
