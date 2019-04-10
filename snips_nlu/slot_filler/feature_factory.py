from __future__ import unicode_literals

from abc import abstractmethod, ABCMeta
from builtins import str

from future.utils import with_metaclass
from snips_nlu_parsers import get_supported_grammar_entities
from snips_nlu_utils import get_shape

from snips_nlu.common.abc_utils import classproperty
from snips_nlu.common.registrable import Registrable
from snips_nlu.constants import (
    CUSTOM_ENTITY_PARSER_USAGE, END, GAZETTEERS, LANGUAGE, RES_MATCH_RANGE,
    START, STEMS, WORD_CLUSTERS, CUSTOM_ENTITY_PARSER, BUILTIN_ENTITY_PARSER,
    RESOURCES)
from snips_nlu.dataset import (
    extract_intent_entities, get_dataset_gazetteer_entities)
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.entity_parser.custom_entity_parser import \
    CustomEntityParserUsage
from snips_nlu.languages import get_default_sep
from snips_nlu.preprocessing import Token, normalize_token, stem_token
from snips_nlu.resources import get_gazetteer, get_word_cluster
from snips_nlu.slot_filler.crf_utils import TaggingScheme, get_scheme_prefix
from snips_nlu.slot_filler.feature import Feature
from snips_nlu.slot_filler.features_utils import (
    entity_filter, get_word_chunk, initial_string_from_tokens)


class CRFFeatureFactory(with_metaclass(ABCMeta, Registrable)):
    """Abstraction to implement to build CRF features

    A :class:`CRFFeatureFactory` is initialized with a dict which describes
    the feature, it must contains the three following keys:

    -   'factory_name'
    -   'args': the parameters of the feature, if any
    -   'offsets': the offsets to consider when using the feature in the CRF.
        An empty list corresponds to no feature.


    In addition, a 'drop_out' to use at training time can be specified.
    """

    def __init__(self, factory_config, **shared):
        self.factory_config = factory_config
        self.resources = shared.get(RESOURCES)
        self.builtin_entity_parser = shared.get(BUILTIN_ENTITY_PARSER)
        self.custom_entity_parser = shared.get(CUSTOM_ENTITY_PARSER)

    @classmethod
    def from_config(cls, factory_config, **shared):
        """Retrieve the :class:`CRFFeatureFactory` corresponding the provided
        config

        Raises:
            NotRegisteredError: when the factory is not registered
        """
        factory_name = factory_config["factory_name"]
        factory = cls.by_name(factory_name)
        return factory(factory_config, **shared)

    @classproperty
    def name(cls):  # pylint:disable=no-self-argument
        return CRFFeatureFactory.registered_name(cls)

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

    def get_required_resources(self):
        return None


class SingleFeatureFactory(with_metaclass(ABCMeta, CRFFeatureFactory)):
    """A CRF feature factory which produces only one feature"""

    @property
    def feature_name(self):
        # by default, use the factory name
        return self.name

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


@CRFFeatureFactory.register("is_digit")
class IsDigitFactory(SingleFeatureFactory):
    """Feature: is the considered token a digit?"""

    def compute_feature(self, tokens, token_index):
        return "1" if tokens[token_index].value.isdigit() else None


@CRFFeatureFactory.register("is_first")
class IsFirstFactory(SingleFeatureFactory):
    """Feature: is the considered token the first in the input?"""

    def compute_feature(self, tokens, token_index):
        return "1" if token_index == 0 else None


@CRFFeatureFactory.register("is_last")
class IsLastFactory(SingleFeatureFactory):
    """Feature: is the considered token the last in the input?"""

    def compute_feature(self, tokens, token_index):
        return "1" if token_index == len(tokens) - 1 else None


@CRFFeatureFactory.register("prefix")
class PrefixFactory(SingleFeatureFactory):
    """Feature: a prefix of the considered token

    This feature has one parameter, *prefix_size*, which specifies the size of
    the prefix
    """

    @property
    def feature_name(self):
        return "prefix_%s" % self.prefix_size

    @property
    def prefix_size(self):
        return self.args["prefix_size"]

    def compute_feature(self, tokens, token_index):
        return get_word_chunk(normalize_token(tokens[token_index]),
                              self.prefix_size, 0)


@CRFFeatureFactory.register("suffix")
class SuffixFactory(SingleFeatureFactory):
    """Feature: a suffix of the considered token

    This feature has one parameter, *suffix_size*, which specifies the size of
    the suffix
    """

    @property
    def feature_name(self):
        return "suffix_%s" % self.suffix_size

    @property
    def suffix_size(self):
        return self.args["suffix_size"]

    def compute_feature(self, tokens, token_index):
        return get_word_chunk(normalize_token(tokens[token_index]),
                              self.suffix_size, len(tokens[token_index].value),
                              reverse=True)


@CRFFeatureFactory.register("length")
class LengthFactory(SingleFeatureFactory):
    """Feature: the length (characters) of the considered token"""

    def compute_feature(self, tokens, token_index):
        return str(len(tokens[token_index].value))


@CRFFeatureFactory.register("ngram")
class NgramFactory(SingleFeatureFactory):
    """Feature: the n-gram consisting of the considered token and potentially
    the following ones

    This feature has several parameters:

    -   'n' (int): Corresponds to the size of the n-gram. n=1 corresponds to a
        unigram, n=2 is a bigram etc
    -   'use_stemming' (bool): Whether or not to stem the n-gram
    -   'common_words_gazetteer_name' (str, optional): If defined, use a
        gazetteer of common words and replace out-of-corpus ngram with the
        alias
        'rare_word'

    """

    def __init__(self, factory_config, **shared):
        super(NgramFactory, self).__init__(factory_config, **shared)
        self.n = self.args["n"]
        if self.n < 1:
            raise ValueError("n should be >= 1")

        self.use_stemming = self.args["use_stemming"]
        self.common_words_gazetteer_name = self.args[
            "common_words_gazetteer_name"]
        self._gazetteer = None
        self._language = None
        self.language = self.args.get("language_code")

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if value is not None:
            self._language = value
            self.args["language_code"] = self.language

    @property
    def gazetteer(self):
        # Load the gazetteer lazily
        if self.common_words_gazetteer_name is None:
            return None
        if self._gazetteer is None:
            self._gazetteer = get_gazetteer(
                self.resources, self.common_words_gazetteer_name)
        return self._gazetteer

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
                    stems = (stem_token(t, self.resources)
                             for t in tokens[token_index:end])
                    return get_default_sep(self.language).join(stems)
                normalized_values = (normalize_token(t)
                                     for t in tokens[token_index:end])
                return get_default_sep(self.language).join(normalized_values)
            words = []
            for t in tokens[token_index:end]:
                if self.use_stemming:
                    value = stem_token(t, self.resources)
                else:
                    value = normalize_token(t)
                words.append(value if value in self.gazetteer else "rare_word")
            return get_default_sep(self.language).join(words)
        return None

    def get_required_resources(self):
        resources = dict()
        if self.common_words_gazetteer_name is not None:
            resources[GAZETTEERS] = {self.common_words_gazetteer_name}
        if self.use_stemming:
            resources[STEMS] = True
        return resources


@CRFFeatureFactory.register("shape_ngram")
class ShapeNgramFactory(SingleFeatureFactory):
    """Feature: the shape of the n-gram consisting of the considered token and
    potentially the following ones

    This feature has one parameters, *n*, which corresponds to the size of the
    n-gram.

    Possible types of shape are:

        -   'xxx' -> lowercased
        -   'Xxx' -> Capitalized
        -   'XXX' -> UPPERCASED
        -   'xX' -> None of the above
    """

    def __init__(self, factory_config, **shared):
        super(ShapeNgramFactory, self).__init__(factory_config, **shared)
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
            self._language = value
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
            return get_default_sep(self.language).join(
                get_shape(t.value) for t in tokens[token_index:end])
        return None


@CRFFeatureFactory.register("word_cluster")
class WordClusterFactory(SingleFeatureFactory):
    """Feature: The cluster which the considered token belongs to, if any

    This feature has several parameters:

    -   'cluster_name' (str): the name of the word cluster to use
    -   'use_stemming' (bool): whether or not to stem the token before looking
        for its cluster

    Typical words clusters are the Brown Clusters in which words are
    clustered into a binary tree resulting in clusters of the form '100111001'
    See https://en.wikipedia.org/wiki/Brown_clustering
    """

    def __init__(self, factory_config, **shared):
        super(WordClusterFactory, self).__init__(factory_config, **shared)
        self.cluster_name = self.args["cluster_name"]
        self.use_stemming = self.args["use_stemming"]
        self._cluster = None

    @property
    def cluster(self):
        if self._cluster is None:
            self._cluster = get_word_cluster(self.resources, self.cluster_name)
        return self._cluster

    @property
    def feature_name(self):
        return "word_cluster_%s" % self.cluster_name

    def compute_feature(self, tokens, token_index):
        if self.use_stemming:
            value = stem_token(tokens[token_index], self.resources)
        else:
            value = normalize_token(tokens[token_index])
        return self.cluster.get(value, None)

    def get_required_resources(self):
        return {
            WORD_CLUSTERS: {self.cluster_name},
            STEMS: self.use_stemming
        }


@CRFFeatureFactory.register("entity_match")
class CustomEntityMatchFactory(CRFFeatureFactory):
    """Features: does the considered token belongs to the values of one of the
    entities in the training dataset

    This factory builds as many features as there are entities in the dataset,
    one per entity.

    It has the following parameters:

    -   'use_stemming' (bool): whether or not to stem the token before looking
        for it among the (stemmed) entity values
    -   'tagging_scheme_code' (int): Represents a :class:`.TaggingScheme`. This
        allows to give more information about the match.
    """

    def __init__(self, factory_config, **shared):
        super(CustomEntityMatchFactory, self).__init__(factory_config,
                                                       **shared)
        self.use_stemming = self.args["use_stemming"]
        self.tagging_scheme = TaggingScheme(
            self.args["tagging_scheme_code"])
        self._entities = None
        self.entities = self.args.get("entities")

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, value):
        if value is not None:
            self._entities = value
            self.args["entities"] = value

    def fit(self, dataset, intent):
        self.entities = extract_intent_entities(
            dataset, lambda e: not is_builtin_entity(e))[intent]
        self.entities = list(self.entities)
        return self

    def _transform(self, tokens):
        if self.use_stemming:
            light_tokens = (stem_token(t, self.resources) for t in tokens)
        else:
            light_tokens = (normalize_token(t) for t in tokens)
        current_index = 0
        transformed_tokens = []
        for light_token in light_tokens:
            transformed_token = Token(
                value=light_token,
                start=current_index,
                end=current_index + len(light_token))
            transformed_tokens.append(transformed_token)
            current_index = transformed_token.end + 1
        return transformed_tokens

    def build_features(self):
        features = []
        for entity_name in self.entities:
            # We need to call this wrapper in order to properly capture
            # `entity_name`
            entity_match = self._build_entity_match_fn(entity_name)

            for offset in self.offsets:
                feature = Feature("entity_match_%s" % entity_name,
                                  entity_match, offset, self.drop_out)
                features.append(feature)
        return features

    def _build_entity_match_fn(self, entity):

        def entity_match(tokens, token_index):
            transformed_tokens = self._transform(tokens)
            text = initial_string_from_tokens(transformed_tokens)
            token_start = transformed_tokens[token_index].start
            token_end = transformed_tokens[token_index].end
            custom_entities = self.custom_entity_parser.parse(
                text, scope=[entity], use_cache=True)
            # only keep builtin entities (of type `entity`) which overlap with
            # the current token
            custom_entities = [ent for ent in custom_entities
                               if entity_filter(ent, token_start, token_end)]
            if custom_entities:
                # In most cases, 0 or 1 entity will be found. We fall back to
                # the first entity if 2 or more were found
                ent = custom_entities[0]
                indexes = []
                for index, token in enumerate(transformed_tokens):
                    if entity_filter(ent, token.start, token.end):
                        indexes.append(index)
                return get_scheme_prefix(token_index, indexes,
                                         self.tagging_scheme)
            return None

        return entity_match

    def get_required_resources(self):
        if self.use_stemming:
            return {
                STEMS: True,
                CUSTOM_ENTITY_PARSER_USAGE: CustomEntityParserUsage.WITH_STEMS
            }
        return {
            STEMS: False,
            CUSTOM_ENTITY_PARSER_USAGE:
                CustomEntityParserUsage.WITHOUT_STEMS
        }


@CRFFeatureFactory.register("builtin_entity_match")
class BuiltinEntityMatchFactory(CRFFeatureFactory):
    """Features: is the considered token part of a builtin entity such as a
    date, a temperature etc

    This factory builds as many features as there are builtin entities
    available in the considered language.

    It has one parameter, *tagging_scheme_code*, which represents a
    :class:`.TaggingScheme`. This allows to give more information about the
    match.
    """

    def __init__(self, factory_config, **shared):
        super(BuiltinEntityMatchFactory, self).__init__(factory_config,
                                                        **shared)
        self.tagging_scheme = TaggingScheme(
            self.args["tagging_scheme_code"])
        self.builtin_entities = None
        self.builtin_entities = self.args.get("entity_labels")
        self._language = None
        self.language = self.args.get("language_code")

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if value is not None:
            self._language = value
            self.args["language_code"] = self.language

    def fit(self, dataset, intent):
        self.language = dataset[LANGUAGE]
        self.builtin_entities = sorted(
            self._get_builtin_entity_scope(dataset, intent))
        self.args["entity_labels"] = self.builtin_entities

    def build_features(self):
        features = []

        for builtin_entity in self.builtin_entities:
            # We need to call this wrapper in order to properly capture
            # `builtin_entity`
            builtin_entity_match = self._build_entity_match_fn(builtin_entity)
            for offset in self.offsets:
                feature_name = "builtin_entity_match_%s" % builtin_entity
                feature = Feature(feature_name, builtin_entity_match, offset,
                                  self.drop_out)
                features.append(feature)

        return features

    def _build_entity_match_fn(self, builtin_entity):

        def builtin_entity_match(tokens, token_index):
            text = initial_string_from_tokens(tokens)
            start = tokens[token_index].start
            end = tokens[token_index].end

            builtin_entities = self.builtin_entity_parser.parse(
                text, scope=[builtin_entity], use_cache=True)
            # only keep builtin entities (of type `builtin_entity`) which
            # overlap with the current token
            builtin_entities = [ent for ent in builtin_entities
                                if entity_filter(ent, start, end)]
            if builtin_entities:
                # In most cases, 0 or 1 entity will be found. We fall back to
                # the first entity if 2 or more were found
                ent = builtin_entities[0]
                entity_start = ent[RES_MATCH_RANGE][START]
                entity_end = ent[RES_MATCH_RANGE][END]
                indexes = []
                for index, token in enumerate(tokens):
                    if (entity_start <= token.start < entity_end) \
                            and (entity_start < token.end <= entity_end):
                        indexes.append(index)
                return get_scheme_prefix(token_index, indexes,
                                         self.tagging_scheme)
            return None

        return builtin_entity_match

    @staticmethod
    def _get_builtin_entity_scope(dataset, intent=None):
        language = dataset[LANGUAGE]
        grammar_entities = list(get_supported_grammar_entities(language))
        gazetteer_entities = list(
            get_dataset_gazetteer_entities(dataset, intent))
        return grammar_entities + gazetteer_entities
