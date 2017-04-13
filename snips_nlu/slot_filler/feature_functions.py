import re
from collections import namedtuple

from crf_resources import get_word_clusters
from snips_nlu.built_in_entities import get_built_in_entities, BuiltInEntity
from snips_nlu.constants import (MATCH_RANGE)
from snips_nlu.languages import Language
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features
from snips_nlu.slot_filler.en.specific_features_functions import \
    language_specific_features as en_features
from snips_nlu.slot_filler.ko.specific_features_functions import \
    language_specific_features as ko_features

TOKEN_NAME = "token"
LOWER_REGEX = re.compile(r"^[^A-Z]+$")
UPPER_REGEX = re.compile(r"^[^a-z]+$")
TITLE_REGEX = re.compile(r"^[A-Z][^A-Z]+$")

BaseFeatureFunction = namedtuple("BaseFeatureFunction", "name function")


def crf_features(intent_entities, language):
    if language == Language.EN:
        return en_features(module_name=__name__,
                           intent_entities=intent_entities)
    elif language == Language.ES:
        return default_features(__name__, language, intent_entities,
                                use_stemming=True,
                                entities_offsets=(-2, -1, 0),
                                entity_keep_prob=.5)
    elif language == Language.FR:
        return default_features(__name__, language, intent_entities,
                                use_stemming=True,
                                entities_offsets=(-2, -1, 0),
                                entity_keep_prob=.5)
    elif language == Language.DE:
        return default_features(__name__, language, intent_entities,
                                use_stemming=True,
                                entities_offsets=(-2, -1, 0),
                                entity_keep_prob=.5)
    elif language == Language.KO:
        return ko_features(module_name=__name__,
                           intent_entities=intent_entities)
    else:
        raise NotImplementedError("Feature function are not implemented for "
                                  "%s" % language)


# Helpers for base feature functions and factories
def char_range_to_token_range(char_range, tokens_as_string):
    start, end = char_range
    # TODO: if possible avoid looping on the tokens for better efficiency
    current_length = 0
    token_start = None
    for i, t in enumerate(tokens_as_string):
        if current_length == start:
            token_start = i
            break
        current_length += len(t) + 1
    if token_start is None:
        return

    token_end = None
    current_length -= 1  # Remove the last space
    for i, t in enumerate(tokens_as_string[token_start:]):
        current_length += len(t) + 1
        if current_length == end:
            token_end = token_start + i + 1
            break
    if token_end is None:
        return
    return token_start, token_end


def get_shape(string):
    if LOWER_REGEX.match(string):
        shape = "xxx"
    elif UPPER_REGEX.match(string):
        shape = "XXX"
    elif TITLE_REGEX.match(string):
        shape = "Xxx"
    else:
        shape = "xX"
    return shape


def get_word_chunk(word, chunk_size, chunk_start, reverse=False):
    if chunk_size < 1:
        raise ValueError("chunk size should be >= 1")
    if chunk_size > len(word):
        return None
    start = chunk_start - chunk_size if reverse else chunk_start
    end = chunk_start if reverse else chunk_start + chunk_size
    return word[start:end]


def initial_string_from_tokens(tokens):
    current_index = 0
    s = ""
    for t in tokens:
        if t.start > current_index:
            s += " " * (t.start - current_index)
        s += t.value
        current_index = t.end
    return s


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
        return get_word_chunk(tokens[token_index].value.lower(), prefix_size,
                              0)

    return BaseFeatureFunction("prefix-%s" % prefix_size, prefix)


def get_suffix_fn(suffix_size):
    def suffix(tokens, token_index):
        return get_word_chunk(tokens[token_index].value.lower(), suffix_size,
                              len(tokens[token_index].value), reverse=True)

    return BaseFeatureFunction("suffix-%s" % suffix_size, suffix)


def get_ngram_fn(n, use_stemming, common_words=None):
    if n < 1:
        raise ValueError("n should be >= 1")

    def ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and 0 < end <= max_len:
            if common_words is None:
                if use_stemming:
                    return " ".join(t.stem.lower()
                                    for t in tokens[token_index:end])
                else:
                    return " ".join(t.value.lower()
                                    for t in tokens[token_index:end])
            else:
                words = []
                for t in tokens[token_index:end]:
                    lowered = t.stem.lower() if use_stemming else \
                        t.value.lower()
                    words.append(lowered if t.value.lower() in common_words
                                 else "rare_word")
                return " ".join(words)
        return None

    return BaseFeatureFunction("ngram_%s" % n, ngram)


def get_shape_ngram_fn(n):
    if n < 1:
        raise ValueError("n should be >= 1")

    def shape_ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and 0 <= end < max_len:
            return " ".join(get_shape(t.value)
                            for t in tokens[token_index:end])
        return None

    return BaseFeatureFunction("shape_ngram_%s" % n, shape_ngram)


def get_word_cluster_fn(cluster_name, language_code):
    language = Language.from_iso_code(language_code)

    def word_cluster(tokens, token_index):
        return get_word_clusters(language)[cluster_name].get(
            tokens[token_index].value.lower(), None)

    return BaseFeatureFunction("word_cluster_%s" % cluster_name, word_cluster)


def get_token_is_in_fn(collection, collection_name, use_stemming):
    lowered_collection = set([c.lower() for c in collection])

    def token_is_in(tokens, token_index):
        token_string = tokens[token_index].stem.lower() if use_stemming \
            else tokens[token_index].value.lower()
        return "1" if token_string in lowered_collection else None

    return BaseFeatureFunction("token_is_in_%s" % collection_name, token_is_in)


def get_is_in_gazetteer_fn(gazetteer_name, max_ngram_size=None):
    def is_in_gazetter(tokens, token_index):
        pass

    return BaseFeatureFunction("is_in_gazetteer_%s" % gazetteer_name,
                               is_in_gazetter)


def get_built_in_annotation_fn(built_in_entity_label, language_code):
    language = Language.from_iso_code(language_code)
    built_in_entity = BuiltInEntity.from_label(built_in_entity_label)
    feature_name = "built-in-%s" % built_in_entity.value["label"]

    def built_in_annotation(tokens, token_index):
        text = initial_string_from_tokens(tokens)

        built_ins = get_built_in_entities(text, language,
                                          scope=[built_in_entity])
        start = tokens[token_index].start
        end = tokens[token_index].end
        for ent in built_ins:
            range_start = ent[MATCH_RANGE][0]
            range_end = ent[MATCH_RANGE][1]
            if (range_start <= start < range_end) \
                    and (range_start < end <= range_end):
                return "1"

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
