from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from builtins import map, object

from future.utils import with_metaclass, iteritems
from nlu_utils import normalize

from snips_nlu.builtin_entities import (
    get_builtin_entities, get_supported_builtin_entities, BuiltInEntity)
from snips_nlu.constants import (
    LANGUAGE, UTTERANCES, TOKEN_INDEXES, NGRAM, RES_MATCH_RANGE)
from snips_nlu.languages import Language
from snips_nlu.preprocessing import stem
from snips_nlu.resources import get_gazetteer, get_word_clusters
from snips_nlu.slot_filler.crf_utils import TaggingScheme, get_scheme_prefix
from snips_nlu.slot_filler.feature import Feature
from snips_nlu.slot_filler.features_utils import (
    get_word_chunk, get_shape, get_all_ngrams, initial_string_from_tokens,
    entity_filter, get_intent_custom_entities)


class CRFFeatureFactory(with_metaclass(ABCMeta, object)):
    """Abstraction to implement to build CRF features

    A :class:`CRFFeatureFactory` is initialized with a dict which describes
    the feature, it must contains the three following keys:

    - 'factory_name'
    - 'args': the parameters of the feature, if any
    - 'offsets': the offsets to consider when using the feature in the CRF.
        An empty list corresponds to no feature.

    In addition, a 'drop_out' to use during train time can be specified.
    """

    def __init__(self, factory_config):
        self.factory_config = factory_config

    @property
    def factory_name(self):
        return self.factory_config["factory_name"]

    @property
    def args(self):
        return self.factory_config["args"]

    @property
    def offsets(self):
        return self.factory_config["offsets"]

    @property
    def drop_out(self):
        return self.factory_config.get("drop_out", 0.0)

    def fit(self, dataset, intent):  # pylint: disable=unused-argument
        """Fit the factory, if needed, with the provided *dataset* and *intent*
        """
        return self

    @abstractmethod
    def build_features(self):
        """Build a list of :class:`.Feature`"""
        pass


class SingleFeatureFactory(with_metaclass(ABCMeta, CRFFeatureFactory)):
    """A CRF feature factory which produces only one feature"""

    @property
    def feature_name(self):
        # by default, use the factory name
        return self.factory_name

    @abstractmethod
    def compute_feature(self, tokens, token_index):
        pass

    def build_features(self):
        return [
            Feature(
                base_name=self.feature_name,
                func=self.compute_feature,
                offset=offset,
                drop_out=self.drop_out) for offset in self.offsets
        ]


class IsDigitFactory(SingleFeatureFactory):
    """Feature: is the considered token a digit?"""

    name = "is_digit"

    def compute_feature(self, tokens, token_index):
        return "1" if tokens[token_index].value.isdigit() else None


class IsFirstFactory(SingleFeatureFactory):
    """Feature: is the considered token the first in the input?"""

    name = "is_first"

    def compute_feature(self, tokens, token_index):
        return "1" if token_index == 0 else None


class IsLastFactory(SingleFeatureFactory):
    """Feature: is the considered token the last in the input?"""

    name = "is_last"

    def compute_feature(self, tokens, token_index):
        return "1" if token_index == len(tokens) - 1 else None


class PrefixFactory(SingleFeatureFactory):
    """Feature: a prefix of the considered token

    This feature has one parameter, *prefix_size*, which specifies the size of
    the prefix
    """

    name = "prefix"

    @property
    def feature_name(self):
        return "prefix_%s" % self.prefix_size

    @property
    def prefix_size(self):
        return self.args["prefix_size"]

    def compute_feature(self, tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              self.prefix_size, 0)


class SuffixFactory(SingleFeatureFactory):
    """Feature: a suffix of the considered token

    This feature has one parameter, *suffix_size*, which specifies the size of
    the suffix
    """

    name = "suffix"

    @property
    def feature_name(self):
        return "suffix_%s" % self.suffix_size

    @property
    def suffix_size(self):
        return self.args["suffix_size"]

    def compute_feature(self, tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              self.suffix_size, len(tokens[token_index].value),
                              reverse=True)


class LengthFactory(SingleFeatureFactory):
    """Feature: the length (characters) of the considered token"""

    name = "length"

    def compute_feature(self, tokens, token_index):
        return len(tokens[token_index].value)


class NgramFactory(SingleFeatureFactory):
    """Feature: the n-gram consisting of the considered token and potentially
        the following ones

    This feature has several parameters:

    - 'n' (int): Corresponds to the size of the n-gram. n=1 corresponds
        to a unigram, n=2 is a bigram etc
    - 'use_stemming' (bool): Whether or not to stem the n-gram
    - 'common_words_gazetteer_name' (str, optional): If defined, use a
        gazetteer of common words and replace out-of-corpus ngram with the
        alias 'rare_word'
    """

    name = "ngram"

    def __init__(self, factory_config):
        super(NgramFactory, self).__init__(factory_config)
        self.n = self.args["n"]
        if self.n < 1:
            raise ValueError("n should be >= 1")

        self.use_stemming = self.args["use_stemming"]
        self.common_words_gazetteer_name = self.args[
            "common_words_gazetteer_name"]
        self._language = None
        self.language = self.args.get("language_code")
        self.gazetteer = None

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if value is not None:
            self._language = Language.from_iso_code(value)
            self.args["language_code"] = self.language.iso_code
            if self.common_words_gazetteer_name is not None:
                gazetteer = get_gazetteer(self.language,
                                          self.common_words_gazetteer_name)
                if self.use_stemming:
                    gazetteer = set(stem(w, self.language) for w in gazetteer)
                self.gazetteer = gazetteer

    @property
    def feature_name(self):
        return "ngram_%s" % self.n

    def fit(self, dataset, intent):
        self.language = dataset[LANGUAGE]

    def compute_feature(self, tokens, token_index):
        max_len = len(tokens)
        end = token_index + self.n
        if 0 <= token_index < max_len and end <= max_len:
            if self.gazetteer is None:
                if self.use_stemming:
                    return self.language.default_sep.join(
                        t.stem for t in tokens[token_index:end])
                return self.language.default_sep.join(
                    t.normalized_value for t in tokens[token_index:end])
            words = []
            for t in tokens[token_index:end]:
                normalized = t.stem if self.use_stemming else \
                    t.normalized_value
                words.append(normalized if normalized in self.gazetteer
                             else "rare_word")
            return self.language.default_sep.join(words)
        return None


class ShapeNgramFactory(SingleFeatureFactory):
    """Feature: the shape of the n-gram consisting of the considered token and
        potentially the following ones

    This feature has one parameters, *n*, which corresponds to the size of the
    n-gram.

    Possible types of shape are:

    - xxx: lowercased
    - Xxx: Capitalized
    - XXX: UPPERCASED
    - xX: anything else
    """

    name = "shape_ngram"

    def __init__(self, factory_config):
        super(ShapeNgramFactory, self).__init__(factory_config)
        self.n = self.args["n"]
        if self.n < 1:
            raise ValueError("n should be >= 1")
        self._language = None
        self.language = self.args.get("language_code")

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if value is not None:
            self._language = Language.from_iso_code(value)
            self.args["language_code"] = value

    @property
    def feature_name(self):
        return "shape_ngram_%s" % self.n

    def fit(self, dataset, intent):
        self.language = dataset[LANGUAGE]

    def compute_feature(self, tokens, token_index):
        max_len = len(tokens)
        end = token_index + self.n
        if 0 <= token_index < max_len and end <= max_len:
            return self.language.default_sep.join(
                get_shape(t.value) for t in tokens[token_index:end])
        return None


class WordClusterFactory(SingleFeatureFactory):
    """Feature: The cluster which the considered token belongs to, if any

    This feature has several parameters:

    - 'cluster_name' (str): the name of the word cluster to use
    - 'use_stemming' (bool): whether or not to stem the token before
        looking for its cluster

    Typical words clusters are the Brown Clusters in which words are
    clustered into a binary tree resulting in clusters of the form '100111001'
    See https://en.wikipedia.org/wiki/Brown_clustering
    """

    name = "word_cluster"

    def __init__(self, factory_config):
        super(WordClusterFactory, self).__init__(factory_config)
        self.cluster_name = self.args["cluster_name"]
        self.cluster = None
        self._language = None
        self.language = self.args.get("language_code")
        self.use_stemming = self.args["use_stemming"]

    @property
    def feature_name(self):
        return "word_cluster_%s" % self.cluster_name

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if value is not None:
            self._language = Language.from_iso_code(value)
            self.cluster = get_word_clusters(self.language)[self.cluster_name]
            self.args["language_code"] = self.language.iso_code

    def fit(self, dataset, intent):
        self.language = dataset[LANGUAGE]

    def compute_feature(self, tokens, token_index):
        normalized_value = tokens[token_index].stem if self.use_stemming \
            else tokens[token_index].normalized_value
        cluster = get_word_clusters(self.language)[self.cluster_name]
        return cluster.get(normalized_value, None)


class EntityMatchFactory(CRFFeatureFactory):
    """Features: does the considered token belongs to the values of one of the
        entities in the training dataset

    This factory builds as many features as there are entities in the dataset,
    one per entity.

    It has the following parameters:

    - 'use_stemming' (bool): whether or not to stem the token before
        looking for it among the (stemmed) entity values
    - 'tagging_scheme_code' (int): Represents a :class:`.TaggingScheme`. This
        allows to give more information about the match.
    """

    name = "entity_match"

    def __init__(self, factory_config):
        super(EntityMatchFactory, self).__init__(factory_config)
        self.use_stemming = self.args["use_stemming"]
        self.tagging_scheme = TaggingScheme(
            self.args["tagging_scheme_code"])
        self.collections = self.args.get("collections")
        self._language = None
        self.language = self.args.get("language_code")

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if value is not None:
            self._language = Language.from_iso_code(value)
            self.args["language_code"] = self.language.iso_code

    def fit(self, dataset, intent):
        self.language = dataset[LANGUAGE]

        def preprocess(string):
            normalized = normalize(string)
            return stem(normalized, self.language) if self.use_stemming \
                else normalized

        intent_entities = get_intent_custom_entities(dataset, intent)
        self.collections = dict()
        for entity_name, entity in iteritems(intent_entities):
            if not entity[UTTERANCES]:
                continue
            collection = list(preprocess(e) for e in entity[UTTERANCES])
            self.collections[entity_name] = collection
        self.args["collections"] = self.collections
        return self

    def _transform(self, token):
        return token.stem if self.use_stemming else token.normalized_value

    def build_features(self):
        features = []
        for name, collection in iteritems(self.collections):
            # We need to call this wrapper in order to properly capture
            # `collection`
            collection_match = self._build_collection_match_fn(collection)

            for offset in self.offsets:
                feature = Feature("entity_match_%s" % name,
                                  collection_match, offset, self.drop_out)
                features.append(feature)
        return features

    def _build_collection_match_fn(self, collection):
        collection_set = set(collection)

        def collection_match(tokens, token_index):
            normalized_tokens = list(map(self._transform, tokens))
            ngrams = get_all_ngrams(normalized_tokens)
            ngrams = [ngram for ngram in ngrams if
                      token_index in ngram[TOKEN_INDEXES]]
            ngrams = sorted(ngrams, key=lambda ng: len(ng[TOKEN_INDEXES]),
                            reverse=True)
            for ngram in ngrams:
                if ngram[NGRAM] in collection_set:
                    return get_scheme_prefix(token_index,
                                             sorted(ngram[TOKEN_INDEXES]),
                                             self.tagging_scheme)
            return None

        return collection_match


class BuiltinEntityMatchFactory(CRFFeatureFactory):
    """Features: is the considered token part of a builtin entity such as a
        date, a temperature etc

    This factory builds as many features as there are builtin entities
    available in the considered language.

    It has one parameter, *tagging_scheme_code*, which represents a
    :class:`.TaggingScheme`. This allows to give more information about the
    match.
    """

    name = "builtin_entity_match"

    def __init__(self, factory_config):
        super(BuiltinEntityMatchFactory, self).__init__(factory_config)
        self.tagging_scheme = TaggingScheme(
            self.args["tagging_scheme_code"])
        self.builtin_entities = None
        entity_labels = self.args.get("entity_labels")
        if entity_labels is not None:
            self.builtin_entities = [BuiltInEntity.from_label(label)
                                     for label in entity_labels]
        self._language = None
        self.language = self.args.get("language_code")

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if value is not None:
            self._language = Language.from_iso_code(value)
            self.args["language_code"] = self.language.iso_code

    def fit(self, dataset, intent):
        self.language = dataset[LANGUAGE]
        self.builtin_entities = get_supported_builtin_entities(self.language)
        self.args["entity_labels"] = [entity.label for entity in
                                      self.builtin_entities]

    def build_features(self):
        features = []

        for builtin_entity in self.builtin_entities:
            # We need to call this wrapper in order to properly capture
            # `builtin_entity`
            builtin_entity_match = self._build_entity_match_fn(builtin_entity)
            for offset in self.offsets:
                feature_name = "builtin_entity_match_%s" % builtin_entity.label
                feature = Feature(feature_name, builtin_entity_match, offset,
                                  self.drop_out)
                features.append(feature)

        return features

    def _build_entity_match_fn(self, builtin_entity):

        def builtin_entity_match(tokens, token_index):
            text = initial_string_from_tokens(tokens)
            start = tokens[token_index].start
            end = tokens[token_index].end

            builtin_entities = get_builtin_entities(
                text, self.language, scope=[builtin_entity])
            builtin_entities = [ent for ent in builtin_entities
                                if entity_filter(ent, start, end)]
            for ent in builtin_entities:
                entity_start = ent[RES_MATCH_RANGE][0]
                entity_end = ent[RES_MATCH_RANGE][1]
                indexes = []
                for index, token in enumerate(tokens):
                    if (entity_start <= token.start < entity_end) \
                            and (entity_start < token.end <= entity_end):
                        indexes.append(index)
                return get_scheme_prefix(token_index, indexes,
                                         self.tagging_scheme)

        return builtin_entity_match


FACTORIES = [IsDigitFactory, IsFirstFactory, IsLastFactory, PrefixFactory,
             SuffixFactory, LengthFactory, NgramFactory, ShapeNgramFactory,
             WordClusterFactory, EntityMatchFactory, BuiltinEntityMatchFactory]


def get_feature_factory(factory_config):
    """Retrieve the :class:`CRFFeatureFactory` corresponding the provided
        config"""
    factory_name = factory_config["factory_name"]
    for factory in FACTORIES:
        if factory_name == factory.name:
            return factory(factory_config)
    raise ValueError("Unknown feature factory: %s" % factory_name)
