from __future__ import unicode_literals

from collections import namedtuple

from snips_nlu.builtin_entities import get_builtin_entities, BuiltInEntity
from snips_nlu.constants import (MATCH_RANGE, TOKEN_INDEXES, NGRAM)
from snips_nlu.languages import Language
from snips_nlu.preprocessing import stem
from snips_nlu.resources import get_word_clusters, get_gazetteer
from snips_nlu.slot_filler.crf_utils import get_scheme_prefix, TaggingScheme
from snips_nlu.slot_filler.de.specific_features_functions import \
    language_specific_features as de_features
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features
from snips_nlu.slot_filler.en.specific_features_functions import \
    language_specific_features as en_features
from snips_nlu.slot_filler.features_utils import get_all_ngrams, get_shape, \
    get_word_chunk, initial_string_from_tokens
from snips_nlu.slot_filler.fr.specific_features_functions import \
    language_specific_features as fr_features
from snips_nlu.slot_filler.ko.specific_features_functions import \
    language_specific_features as ko_features

TOKEN_NAME = "token"

BaseFeatureFunction = namedtuple("BaseFeatureFunction", "name function")


def crf_features(dataset, intent, language, config):
    if language == Language.EN:
        return en_features(dataset, intent, config)
    elif language == Language.ES:
        return default_features(language, dataset, intent, config,
                                use_stemming=True)
    elif language == Language.FR:
        return fr_features(dataset, intent, config)
    elif language == Language.DE:
        return de_features(dataset, intent, config)
    elif language == Language.KO:
        return ko_features(dataset, intent, config)
    else:
        raise NotImplementedError("Feature function are not implemented for "
                                  "%s" % language)


# Base feature functions and factories
def is_digit():
    return BaseFeatureFunction(
        "is_digit",
        lambda tokens, token_index: "1" if tokens[
            token_index].value.isdigit() else None
    )


def is_first():
    return BaseFeatureFunction(
        "is_first",
        lambda tokens, token_index: "1" if token_index == 0 else None
    )


def is_last():
    return BaseFeatureFunction(
        "is_last",
        lambda tokens, token_index: "1" if token_index == len(
            tokens) - 1 else None
    )


def get_prefix_fn(prefix_size):
    def prefix(tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              prefix_size, 0)

    return BaseFeatureFunction("prefix-%s" % prefix_size, prefix)


def get_suffix_fn(suffix_size):
    def suffix(tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              suffix_size, len(tokens[token_index].value),
                              reverse=True)

    return BaseFeatureFunction("suffix-%s" % suffix_size, suffix)


def get_length_fn():
    return BaseFeatureFunction(
        "length", lambda tokens, token_index: len(tokens[token_index].value))


def get_ngram_fn(n, use_stemming, language_code,
                 common_words_gazetteer_name=None):
    language = Language.from_iso_code(language_code)
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
                else:
                    return language.default_sep.join(
                        t.normalized_value for t in tokens[token_index:end])
            else:
                words = []
                for t in tokens[token_index:end]:
                    normalized = t.stem if use_stemming else t.normalized_value
                    words.append(normalized if normalized in gazetteer
                                 else "rare_word")
                return language.default_sep.join(words)
        return None

    return BaseFeatureFunction("ngram_%s" % n, ngram)


def get_shape_ngram_fn(n, language_code):
    language = Language.from_iso_code(language_code)
    if n < 1:
        raise ValueError("n should be >= 1")

    def shape_ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and end <= max_len:
            return language.default_sep.join(get_shape(t.value)
                                             for t in tokens[token_index:end])
        return None

    return BaseFeatureFunction("shape_ngram_%s" % n, shape_ngram)


def get_word_cluster_fn(cluster_name, language_code, use_stemming):
    language = Language.from_iso_code(language_code)

    def word_cluster(tokens, token_index):
        normalized_value = tokens[token_index].stem if use_stemming \
            else tokens[token_index].normalized_value
        return get_word_clusters(language)[cluster_name].get(normalized_value,
                                                             None)

    return BaseFeatureFunction("word_cluster_%s" % cluster_name, word_cluster)


def get_token_is_in_fn(tokens_collection, collection_name, use_stemming,
                       tagging_scheme_code, language_code):
    tagging_scheme = TaggingScheme(tagging_scheme_code)
    tokens_collection = set(tokens_collection)

    def transform(token):
        return token.stem if use_stemming else token.normalized_value

    def token_is_in(tokens, token_index):
        normalized_tokens = map(transform, tokens)
        ngrams = get_all_ngrams(normalized_tokens)
        ngrams = filter(lambda ng: token_index in ng[TOKEN_INDEXES], ngrams)
        ngrams = sorted(ngrams, key=lambda ng: len(ng[TOKEN_INDEXES]),
                        reverse=True)
        for ngram in ngrams:
            if ngram[NGRAM] in tokens_collection:
                return get_scheme_prefix(token_index,
                                         sorted(ngram[TOKEN_INDEXES]),
                                         tagging_scheme)
        return None

    return BaseFeatureFunction("token_is_in_%s" % collection_name, token_is_in)


def get_is_in_gazetteer_fn(gazetteer_name, language_code, tagging_scheme_code,
                           use_stemming):
    language = Language.from_iso_code(language_code)
    gazetteer = get_gazetteer(language, gazetteer_name)
    if use_stemming:
        gazetteer = set(stem(w, language) for w in gazetteer)
    tagging_scheme = TaggingScheme(tagging_scheme_code)

    def transform(token):
        return token.stem if use_stemming else token.normalized_value

    def is_in_gazetter(tokens, token_index):
        normalized_tokens = map(transform, tokens)
        ngrams = get_all_ngrams(normalized_tokens)
        ngrams = filter(lambda ng: token_index in ng[TOKEN_INDEXES], ngrams)
        ngrams = sorted(ngrams, key=lambda ng: len(ng[TOKEN_INDEXES]),
                        reverse=True)
        for ngram in ngrams:
            if ngram[NGRAM] in gazetteer:
                return get_scheme_prefix(token_index,
                                         sorted(ngram[TOKEN_INDEXES]),
                                         tagging_scheme)
        return None

    return BaseFeatureFunction("is_in_gazetteer_%s" % gazetteer_name,
                               is_in_gazetter)


def get_built_in_annotation_fn(built_in_entity_label, language_code,
                               tagging_scheme_code):
    language = Language.from_iso_code(language_code)
    tagging_scheme = TaggingScheme(tagging_scheme_code)
    built_in_entity = BuiltInEntity.from_label(built_in_entity_label)
    feature_name = "built-in-%s" % built_in_entity.value["label"]

    def built_in_annotation(tokens, token_index):
        text = initial_string_from_tokens(tokens)
        start = tokens[token_index].start
        end = tokens[token_index].end

        builtin_entities = get_builtin_entities(text, language,
                                                scope=[built_in_entity])

        builtin_entities = filter(
            lambda _ent: (_ent[MATCH_RANGE][0] <= start < _ent[MATCH_RANGE][
                1]) and (_ent[MATCH_RANGE][0] < end <= _ent[MATCH_RANGE][1]),
            builtin_entities)

        for ent in builtin_entities:
            entity_start = ent[MATCH_RANGE][0]
            entity_end = ent[MATCH_RANGE][1]
            indexes = []
            for index, token in enumerate(tokens):
                if (entity_start <= token.start < entity_end) \
                        and (entity_start < token.end <= entity_end):
                    indexes.append(index)
            return get_scheme_prefix(token_index, indexes, tagging_scheme)

    return BaseFeatureFunction(feature_name, built_in_annotation)


def create_feature_function(base_feature_fn, offset):
    """
    Transforms a base feature function into a feature function 
    """
    if base_feature_fn.name == TOKEN_NAME:
        raise ValueError("'%s' name is reserved" % TOKEN_NAME)
    if offset > 0:
        index = "[+%s]" % offset
    elif offset < 0:
        index = "[%s]" % offset
    else:
        index = ""

    feature_name = base_feature_fn.name + index

    def feature_fn(token_index, cache):
        if not 0 <= (token_index + offset) < len(cache):
            return

        if base_feature_fn.name in cache[token_index + offset]:
            return cache[token_index + offset].get(base_feature_fn.name, None)
        else:
            tokens = [c["token"] for c in cache]
            value = base_feature_fn.function(tokens, token_index + offset)
            if value is not None:
                cache[token_index + offset][base_feature_fn.name] = value
            return value

    return feature_name, feature_fn
