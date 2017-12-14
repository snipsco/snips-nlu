from __future__ import unicode_literals

from nlu_utils import compute_all_ngrams

from snips_nlu.constants import MATCH_RANGE
from snips_nlu.utils import LimitedSizeDict

_NGRAMS_CACHE = LimitedSizeDict(size_limit=1000)


def get_all_ngrams(tokens):
    key = "<||>".join(tokens)
    if key not in _NGRAMS_CACHE:
        ngrams = compute_all_ngrams(tokens, len(tokens))
        _NGRAMS_CACHE[key] = ngrams
    return _NGRAMS_CACHE[key]


def get_shape(string):
    if string.islower():
        shape = "xxx"
    elif string.isupper():
        shape = "XXX"
    # Hack for Rust parallelism (we could use istitle but it does not exists)
    elif len(string) > 1 and string[0].isupper() and string[1:].islower():
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


def entity_filter(entity, start, end):
    return (entity[MATCH_RANGE][0] <= start < entity[MATCH_RANGE][1]) and \
           (entity[MATCH_RANGE][0] < end <= entity[MATCH_RANGE][1])
