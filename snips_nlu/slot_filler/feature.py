from __future__ import unicode_literals

from nlu_utils import normalize

from snips_nlu.builtin_entities import get_builtin_entities, BuiltInEntity
from snips_nlu.constants import (MATCH_RANGE, TOKEN_INDEXES, NGRAM, LANGUAGE,
                                 UTTERANCES)
from snips_nlu.languages import Language
from snips_nlu.preprocessing import stem
from snips_nlu.resources import get_word_clusters, get_gazetteer
from snips_nlu.slot_filler.crf_utils import get_scheme_prefix, TaggingScheme
from snips_nlu.slot_filler.default.default_features_functions import \
    get_intent_custom_entities
from snips_nlu.slot_filler.features.de import features as de_features
from snips_nlu.slot_filler.features.en import features as en_features
from snips_nlu.slot_filler.features.es import features as es_features
from snips_nlu.slot_filler.features.fr import features as fr_features
from snips_nlu.slot_filler.features.ko import features as ko_features
from snips_nlu.slot_filler.features_utils import get_all_ngrams, get_shape, \
    get_word_chunk, initial_string_from_tokens, entity_filter

TOKEN_NAME = "token"


class Feature(object):
    def __init__(self, base_name, func, offset=0, drop_out=0):
        if base_name == TOKEN_NAME:
            raise ValueError("'%s' name is reserved" % TOKEN_NAME)
        self.offset = offset
        self._name = None
        self._base_name = None
        self.base_name = base_name
        self.function = func
        self.drop_out = drop_out

    @property
    def name(self):
        return self._name

    @property
    def base_name(self):
        return self._base_name

    @base_name.setter
    def base_name(self, value):
        self._name = _offset_name(value, self.offset)
        self._base_name = _offset_name(value, 0)

    def compute(self, token_index, cache):
        if not 0 <= (token_index + self.offset) < len(cache):
            return

        if self.base_name in cache[token_index + self.offset]:
            return cache[token_index + self.offset][self.base_name]

        tokens = [c["token"] for c in cache]
        value = self.function(tokens, token_index + self.offset)
        cache[token_index + self.offset][self.base_name] = value
        return value


def _offset_name(name, offset):
    if offset > 0:
        return "%s[+%s]" % (name, offset)
    if offset < 0:
        return "%s[%s]" % (name, offset)
    return name


def crf_features(language):
    if language == Language.EN:
        return en_features
    elif language == Language.ES:
        return es_features
    elif language == Language.FR:
        return fr_features
    elif language == Language.DE:
        return de_features
    elif language == Language.KO:
        return ko_features
    else:
        raise NotImplementedError("Feature function are not implemented for "
                                  "%s" % language)


# Base feature functions and factories
def is_digit():
    return [
        Feature(
            "is_digit",
            lambda tokens, token_index: "1" if tokens[
                token_index].value.isdigit() else None
        )
    ]


def is_first():
    return [
        Feature(
            "is_first",
            lambda tokens, token_index: "1" if token_index == 0 else None
        )
    ]


def is_last():
    return [
        Feature(
            "is_last",
            lambda tokens, token_index: "1" if token_index == len(
                tokens) - 1 else None
        )
    ]


def get_prefix_fn(prefix_size):
    def prefix(tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              prefix_size, 0)

    return [Feature("prefix-%s" % prefix_size, prefix)]


def get_suffix_fn(suffix_size):
    def suffix(tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              suffix_size, len(tokens[token_index].value),
                              reverse=True)

    return [Feature("suffix-%s" % suffix_size, suffix)]


def get_length_fn():
    return [Feature(
        "length", lambda tokens, token_index: len(tokens[token_index].value))]


def get_ngram_fn(n, use_stemming, dataset, common_words_gazetteer_name=None):
    language = Language.from_iso_code(dataset[LANGUAGE])
    if n < 1:
        raise ValueError("n should be >= 1")

    gazetteer = None
    if common_words_gazetteer_name is not None:
        gazetteer = get_gazetteer(language, common_words_gazetteer_name)
        if use_stemming:
            gazetteer = set(stem(w, language) for w in gazetteer)

    def ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and end <= max_len:
            if gazetteer is None:
                if use_stemming:
                    return language.default_sep.join(
                        t.stem for t in tokens[token_index:end])
                return language.default_sep.join(
                    t.normalized_value for t in tokens[token_index:end])
            words = []
            for t in tokens[token_index:end]:
                normalized = t.stem if use_stemming else t.normalized_value
                words.append(normalized if normalized in gazetteer
                             else "rare_word")
            return language.default_sep.join(words)
        return None

    return [Feature("ngram_%s" % n, ngram)]


def get_shape_ngram_fn(n, dataset):
    language = Language.from_iso_code(dataset[LANGUAGE])
    if n < 1:
        raise ValueError("n should be >= 1")

    def shape_ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and end <= max_len:
            return language.default_sep.join(get_shape(t.value)
                                             for t in tokens[token_index:end])
        return None

    return [Feature("shape_ngram_%s" % n, shape_ngram)]


def get_word_cluster_fn(cluster_name, use_stemming, dataset):
    language = Language.from_iso_code(dataset[LANGUAGE])

    def word_cluster(tokens, token_index):
        normalized_value = tokens[token_index].stem if use_stemming \
            else tokens[token_index].normalized_value
        return get_word_clusters(language)[cluster_name].get(normalized_value,
                                                             None)

    return [Feature("word_cluster_%s" % cluster_name, word_cluster)]


# pylint: disable=unused-argument
def get_collection_match_fn(use_stemming, tagging_scheme_code, dataset,
                            intent):
    language = Language.from_iso_code(dataset[LANGUAGE])

    # Entity lookup
    def preprocess(string):
        normalized = normalize(string)
        return stem(normalized, language) if use_stemming else normalized

    def transform(token):
        return token.stem if use_stemming else token.normalized_value

    intent_entities = get_intent_custom_entities(dataset, intent)
    features = []
    for entity_name, entity in intent_entities.iteritems():
        if not entity[UTTERANCES]:
            continue

        collection = list(
            set(preprocess(e) for e in entity[UTTERANCES].keys()))

        tagging_scheme = TaggingScheme(tagging_scheme_code)
        tokens_collection = set(collection)

        def token_is_in(tokens, token_index):
            normalized_tokens = map(transform, tokens)
            ngrams = get_all_ngrams(normalized_tokens)
            ngrams = [ng for ng in ngrams if token_index in ng[TOKEN_INDEXES]]
            ngrams = sorted(ngrams,
                            key=lambda _ngram: len(_ngram[TOKEN_INDEXES]),
                            reverse=True)
            for ngram in ngrams:
                if ngram[NGRAM] in tokens_collection:
                    return get_scheme_prefix(token_index,
                                             sorted(ngram[TOKEN_INDEXES]),
                                             tagging_scheme)
            return None

        feature = Feature("token_is_in_%s" % entity_name, token_is_in)
        features.append(feature)
    return features


# pylint: enable=unused-argument


def get_built_in_annotation_fn(built_in_entity_label, dataset,
                               tagging_scheme_code):
    language = Language.from_iso_code(dataset[LANGUAGE])
    tagging_scheme = TaggingScheme(tagging_scheme_code)
    built_in_entity = BuiltInEntity.from_label(built_in_entity_label)
    feature_name = "built-in-%s" % built_in_entity.value["label"]

    def built_in_annotation(tokens, token_index):
        text = initial_string_from_tokens(tokens)
        start = tokens[token_index].start
        end = tokens[token_index].end

        builtin_entities = get_builtin_entities(
            text, language, scope=[built_in_entity])
        builtin_entities = [ent for ent in builtin_entities
                            if entity_filter(ent, start, end)]
        for ent in builtin_entities:
            entity_start = ent[MATCH_RANGE][0]
            entity_end = ent[MATCH_RANGE][1]
            indexes = []
            for index, token in enumerate(tokens):
                if (entity_start <= token.start < entity_end) \
                        and (entity_start < token.end <= entity_end):
                    indexes.append(index)
            return get_scheme_prefix(token_index, indexes, tagging_scheme)

    return [Feature(feature_name, built_in_annotation)]
