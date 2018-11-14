from __future__ import division, unicode_literals

from collections import Counter, defaultdict

import numpy as np
from future.builtins import object, range, str, zip
from future.utils import iteritems
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import cosine
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_is_fitted
from snips_nlu_utils import normalize
from snips_nlu_utils._snips_nlu_utils_py import compute_all_ngrams

from snips_nlu.constants import (BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER,
                                 CUSTOM_ENTITY_PARSER_USAGE, DATA, END,
                                 ENTITIES, ENTITY, ENTITY_KIND, NGRAM,
                                 RES_MATCH_RANGE, START, TEXT, VALUE)
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.entity_parser.builtin_entity_parser import (BuiltinEntityParser,
                                                           is_builtin_entity)
from snips_nlu.entity_parser.custom_entity_parser import CustomEntityParser
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.preprocessing import stem, tokenize_light
from snips_nlu.resources import (get_stop_words, get_word_cluster)
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

        # self.kmeans_count_vectorizer = CountVectorizer(
        #     tokenizer=lambda x: tokenize_light(x, language),
        #     ngram_range=(1, 2))

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

        normalized_stemmed_utterances = [
            _normalize_stem(
                get_text_from_chunks(u[DATA]),
                self.language,
                self.config.use_stemming
            ) for u in utterances
        ]

        none_class = max(classes)
        utterances_per_classes = defaultdict(list)
        for u, c in zip(normalized_stemmed_utterances, classes):
            if c == none_class:
                continue
            utterances_per_classes[c].append(u)

        k = 5
        self.top_bigrams_per_classes = dict()
        for cls, utt in iteritems(utterances_per_classes):
            bigrams = [
                ng for u in utt
                for ng in compute_all_ngrams(
                    tokenize_light(u, self.language), 2)
            ]
            bigrams = [ng["ngram"] for ng in bigrams
                       if len(ng["token_indexes"]) == 2]
            top_bigrams = Counter(bigrams).most_common(k)
            # print "utterances examples: %s" % utt[:5]
            # print "top bigrams for %s: %s" % (cls, top_bigrams)
            self.top_bigrams_per_classes[cls] = set(top_bigrams)

        X_top_ngrams = self._get_top_ngrams_feature(
            [get_text_from_chunks(u[DATA]) for u in utterances])

        # self.kmeans_count_vectorizer.fit(k_mean_utterances)
        #
        # unique_classes = set(classes)
        # kmeans_utterances_per_class = {c: [] for c in unique_classes}
        # for u, c in zip(k_mean_utterances, classes):
        #     kmeans_utterances_per_class[c].append(u)

        # target_num_clusters = 5
        # self.clusterers = dict()
        # self.normalizers = dict()
        # for c in unique_classes:
        #     c_utterances = kmeans_utterances_per_class[c]
        #     x_clusters = self.kmeans_count_vectorizer.transform(c_utterances)
        #
        #     # Normalize the vectors to compute k-mean with cosine distance
        #     normalizer = Normalizer()
        #     x_clusters = normalizer.fit_transform(x_clusters)
        #     self.normalizers[c] = normalizer
        #
        #     if len(c_utterances) > 5 * target_num_clusters:
        #         n_clusters = target_num_clusters
        #     else:
        #         n_clusters = 1
        #     self.clusterers[c] = KMeans(
        #         n_clusters=n_clusters, n_jobs=-1).fit(x_clusters)
        #
        #
        # X_intent_clusters = self._intents_clusters_features(
        #     k_mean_utterances)

        preprocessed_utterances = self.preprocess_utterances(utterances)
        # pylint: disable=C0103
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(
            preprocessed_utterances)

        # pylint: enable=C0103
        features_idx = {self.tfidf_vectorizer.vocabulary_[word]: word for
                        word
                        in self.tfidf_vectorizer.vocabulary_}

        stop_words = get_stop_words(self.language)

        self.entities = {e: i for i, e in enumerate(dataset[ENTITIES])}

        # X_train = hstack(
        #     (X_train_tfidf, csr_matrix(X_intent_clusters)), format="csr")

        X_train = X_train_tfidf

        # X_train = hstack(
        #     (X_train_tfidf, csr_matrix(X_top_ngrams)), format="csr")

        _, pval = chi2(X_train, classes)
        self.best_features = [i for i, v in enumerate(pval) if
                              v < self.config.pvalue_threshold]

        if not self.best_features:
            self.best_features = [idx for idx, val in enumerate(pval) if
                                  val == pval.min()]

        feature_names = {}
        for utterance_index in self.best_features:
            if utterance_index not in features_idx:
                continue
            feature_names[utterance_index] = {
                "word": features_idx[utterance_index],
                "pval": pval[utterance_index]
            }

        for feat in feature_names:
            if feature_names[feat]["word"] in stop_words:
                if feature_names[feat]["pval"] > \
                        self.config.pvalue_threshold / 2.0:
                    self.best_features.remove(feat)

        # min_cluster_ix = X_train_tfidf.shape[1]
        # max_cluster_ix = min_cluster_ix + X_train_clusters.shape[1]
        # num_entities
        # max_cluster_ix = min_cluster_ix + num_entities
        # for i in range(X_train_tfidf.shape[0], max_cluster_ix):
        #     self.best_features.append(i)
        min_top_k_ix = X_train.shape[1]
        max_top_k_ix = min_top_k_ix + X_top_ngrams.shape[1]
        for i in range(min_top_k_ix, max_top_k_ix):
            self.best_features.append(i)
        return self

    def transform(self, utterances):
        top_k_utterances = [
            _normalize_stem(
                get_text_from_chunks(u[DATA]),
                self.language,
                self.config.use_stemming
            ) for u in utterances
        ]
        # X_clusters = self._intents_clusters_features(k_mean_utterances)

        preprocessed_utterances = self.preprocess_utterances(utterances)
        # # pylint: disable=C0103
        X_tfidf = self.tfidf_vectorizer.transform(
            preprocessed_utterances)
        #
        # X_train = hstack((X_tfidf, csr_matrix(X_clusters)), format="csr")

        X_top_k = self._get_top_ngrams_feature(top_k_utterances)

        X_train = hstack((X_tfidf, csr_matrix(X_top_k)), format="csr")
        X = X_train[:, self.best_features]
        # pylint: enable=C0103
        return X

    def _get_top_ngrams_feature(self, utterances):
        X_top_bigrams = np.zeros(
            (len(utterances), len(self.top_bigrams_per_classes)),
            dtype=np.float)

        normalized_stemmed_utterances = [
            _normalize_stem(u,self.language,self.config.use_stemming)
            for u in utterances
        ]

        for i, u in enumerate(normalized_stemmed_utterances):
            bigrams = [ng["ngram"] for ng in
                       compute_all_ngrams(tokenize_light(u, self.language), 2)
                       if len(ng["token_indexes"]) == 2]
            bigrams = set(bigrams)
            for c, c_bigrams in sorted(
                    iteritems(self.top_bigrams_per_classes)):
                for b in c_bigrams:
                    if b in bigrams:
                        X_top_bigrams[i, c] += 1.0

        return X_top_bigrams

    def _intents_clusters_features(self, utterances):
        x_clusters = []
        for c, clusterer in sorted(iteritems(self.clusterers)):
            intent_centroids = clusterer.cluster_centers_
            normalizer = self.normalizers[c]
            x = self.kmeans_count_vectorizer.transform(utterances)
            x = normalizer.transform(x)
            # Pickup the centroid index for the vectors
            y = clusterer.predict(x)
            x_centroids = [intent_centroids[label] for label in y]
            # Compute the distance to the centroid
            distances = [cosine(vector.todense(), centroid)
                         for vector, centroid in zip(x, x_centroids)]
            x_clusters.append(distances)
        x_clusters = np.array(x_clusters).reshape((len(utterances), -1))
        return np.nan_to_num(x_clusters)

    def _entities_features(self, utterances):
        X_entities = np.zeros((len(utterances), len(self.entities)),
                              dtype=np.float)
        for i, u in enumerate(utterances):
            utterance_tokens = tokenize_light(str(u), self.language)
            normalized_stemmed_tokens = [
                _normalize_stem(t, self.language, self.config.use_stemming)
                for t in utterance_tokens]
            custom_entities = self.custom_entity_parser.parse(
                " ".join(normalized_stemmed_tokens))

            builtin_entities = self.custom_entity_parser.parse(
                " ".join(normalized_stemmed_tokens))
            entities = custom_entities + builtin_entities
            for ent in entities:
                ent_ix = self.entities[ent[ENTITY_KIND]]
                X_entities[i, ent_ix] += 1.0
        return X_entities

    def fit_transform(self, dataset, queries, y):
        return self.fit(dataset, queries, y).transform(queries)

    def preprocess_utterances(self, utterances):
        return [
            _preprocess_utterance(
                u, self.language, self.builtin_entity_parser,
                self.custom_entity_parser,
                self.unknown_words_replacement_string,
                self.config.word_clusters_name,
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


def _preprocess_utterance(utterance, language, builtin_entity_parser,
                          custom_entity_parser, unknownword_replacement_string,
                          word_clusters_name,
                          use_stemming):
    utterance_text = get_text_from_chunks(utterance[DATA])
    utterance_tokens = tokenize_light(utterance_text, language)
    word_clusters_features = _get_word_cluster_features(
        utterance_tokens, word_clusters_name, language)
    normalized_stemmed_tokens = [_normalize_stem(t, language, use_stemming)
                                 for t in utterance_tokens]

    custom_entities = custom_entity_parser.parse(
        " ".join(normalized_stemmed_tokens))
    custom_entities = [e for e in custom_entities
                       if e["value"] != unknownword_replacement_string]
    custom_entities_features = [
        _entity_name_to_feature(e[ENTITY_KIND], language)
        for e in custom_entities]

    builtin_entities = builtin_entity_parser.parse(
        utterance_text, use_cache=True)
    builtin_entities_features = [
        _builtin_entity_to_feature(ent[ENTITY_KIND], language)
        for ent in builtin_entities
    ]

    # We remove values of builtin slots from the utterance to avoid learning
    # specific samples such as '42' or 'tomorrow'
    filtered_normalized_stemmed_tokens = [
        _normalize_stem(chunk[TEXT], language, use_stemming)
        for chunk in utterance[DATA]
        if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
    ]

    features = get_default_sep(language).join(
        filtered_normalized_stemmed_tokens)
    if builtin_entities_features:
        features += " " + " ".join(sorted(builtin_entities_features))
    if custom_entities_features:
        features += " " + " ".join(sorted(custom_entities_features))
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
