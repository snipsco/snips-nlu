import re
from collections import namedtuple

from crf_resources import get_word_clusters
from crf_utils import (UNIT_PREFIX, BEGINNING_PREFIX, LAST_PREFIX,
                       INSIDE_PREFIX)

TOKEN_NAME = "token"
LOWER_REGEX = re.compile(r"^[^A-Z]+$")
UPPER_REGEX = re.compile(r"^[^a-z]+$")
TITLE_REGEX = re.compile(r"^[A-Z][^A-Z]+$")

BaseFeatureFunction = namedtuple("BaseFeatureFunction", "name function")


def default_features():
    features = []
    # n-grams
    features.append((get_ngram_fn(1, common_words=None), -2))
    features.append((get_ngram_fn(1, common_words=None), -1))
    features.append((get_ngram_fn(1, common_words=None), +0))
    features.append((get_ngram_fn(1, common_words=None), +1))
    features.append((get_ngram_fn(1, common_words=None), +2))

    features.append((get_ngram_fn(2, common_words=None), -2))
    features.append((get_ngram_fn(2, common_words=None), +1))

    # Shape
    features.append((get_shape_ngram_fn(1), 0))

    features.append((get_shape_ngram_fn(2), -1))
    features.append((get_shape_ngram_fn(2), 0))

    features.append((get_shape_ngram_fn(3), -1))

    # Digit
    features.append((is_digit, -1))
    features.append((is_digit, 0))
    features.append((is_digit, +1))

    return [create_feature_function(f, offset) for f, offset in features]


# Helpers for base feature functions and factories


def char_range_to_token_range(char_range, tokens):
    start, end = char_range
    # TODO: if possible avoid looping on the tokens for better efficiency
    current_length = 0
    token_start = None
    for i, t in enumerate(tokens):
        if current_length == start:
            token_start = i
            break
        current_length += len(t) + 1
    if token_start is None:
        return

    token_end = None
    current_length -= 1  # Remove the last space
    for i, t in enumerate(tokens[token_start:]):
        current_length += len(t) + 1
        if current_length == end:
            token_end = token_start + i + 1
            break
    if token_end is None:
        return
    return token_start, token_end


def get_shape(token):
    if LOWER_REGEX.match(token):
        shape = "xxx"
    elif UPPER_REGEX.match(token):
        shape = "XXX"
    elif TITLE_REGEX.match(token):
        shape = "Xxx"
    else:
        shape = "xX"
    return shape


def get_word_chunk(word, chunk_size, chunk_start, reverse=False):
    if chunk_size > len(word):
        return None
    start = chunk_start - chunk_size if reverse else chunk_start
    end = chunk_start if reverse else chunk_start + chunk_size
    return word[start:end]


# Base feature functions and factories
is_digit = BaseFeatureFunction(
    "is_digit",
    lambda tokens, token_index: str(int(tokens[token_index].is_digit()))
)

is_first = BaseFeatureFunction(
    "is_first",
    lambda tokens, token_index: str(int(token_index == 0))
)

is_last = BaseFeatureFunction(
    "is_last",
    lambda tokens, token_index: str(int(token_index == len(tokens) - 1))
)


def get_prefix_fn(prefix_size):
    def prefix(tokens, token_index):
        return get_word_chunk(tokens[token_index].lower(), prefix_size, 0)

    return BaseFeatureFunction("prefix-%s" % prefix_size, prefix)


def get_suffix_fn(suffix_size):
    def suffix(tokens, token_index):
        return get_word_chunk(tokens[token_index].lower(), suffix_size,
                              len(tokens[token_index]), reverse=True)

    return BaseFeatureFunction("suffix-%s" % suffix_size, suffix)


def get_ngram_fn(n, common_words=None):
    if n < 1:
        raise ValueError("n should be >= 1")

    def ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and 0 < end <= max_len:
            if common_words is None:
                return " ".join(t.lower() for t in tokens[token_index:end])
            else:
                words = []
                for w in tokens[token_index:end]:
                    lowered = w.lower()
                    words.append(
                        lowered if lowered in common_words else "rare_word")
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
            return " ".join(get_shape(t) for t in tokens[token_index:end])
        return None

    return BaseFeatureFunction("shape_ngram_%s" % n, shape_ngram)


def get_word_cluster_fn(cluster_name):
    def word_cluster(tokens, token_index):
        return get_word_clusters()[cluster_name].get(
            tokens[token_index].lower(), None)

    return BaseFeatureFunction("word_cluster_%s" % cluster_name, word_cluster)


def get_token_is_in(collection, collection_name):
    lowered_collection = set([c.lower() for c in collection])

    def token_is_in(tokens, token_index):
        return str(int(tokens[token_index].lower() in lowered_collection))

    return BaseFeatureFunction("token_is_in_%s" % collection_name, token_is_in)


def get_regex_match_fn(regex, match_name, use_bilou=False):
    def regex_match(tokens, token_index):
        text = " ".join(tokens).lower()

        match = regex.search(text)
        if match is None:
            return

        token_start = len(" ".join(tokens[:token_index])) + 1
        token_end = token_start + len(tokens[token_index])

        match_start = match.start()
        match_end = match.end()
        if not (match_start <= token_start <= match_end
                and match_start <= token_end <= match_end):
            return

        match_token_range = char_range_to_token_range(
            (match_start, match_end), tokens)
        if match_token_range is None:
            return
        match_token_start, match_token_end = match_token_range

        token_position_in_match = token_index - match_token_start

        if token_position_in_match == 0:
            if use_bilou and token_index == match_token_end - 1:
                feature = UNIT_PREFIX + match_name
            else:
                feature = BEGINNING_PREFIX + match_name
        elif token_index == match_token_end - 1:
            if use_bilou:
                feature = LAST_PREFIX + match_name
            else:
                feature = INSIDE_PREFIX + match_name
        else:
            feature = INSIDE_PREFIX + match_name

        return feature

    return BaseFeatureFunction("match_%s" % match_name, regex_match)


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
