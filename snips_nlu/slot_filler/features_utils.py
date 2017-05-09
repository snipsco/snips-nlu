import re
from copy import copy

from snips_nlu.constants import NGRAM, TOKEN_INDEXES
from snips_nlu.utils import LimitedSizeDict

_NGRAMS_CACHE = LimitedSizeDict(size_limit=1000)


def get_all_ngrams(tokens):
    key = "<||>".join(tokens)
    if key not in _NGRAMS_CACHE:
        ngrams = _get_all_ngrams(tokens)
        _NGRAMS_CACHE[key] = ngrams
    return _NGRAMS_CACHE[key]


def _get_all_ngrams(tokens):
    max_ngram_size = len(tokens)
    max_start = len(tokens) - 1
    min_end = 0

    ngrams = []
    for start in xrange(max_start + 1):
        local_ngrams = []
        last_ngram_item = None
        _min_end = max(start, min_end)
        _max_end = min(len(tokens), _min_end + max_ngram_size)
        for end in xrange(_min_end, _max_end):
            if last_ngram_item is not None:
                indexes = copy(last_ngram_item[TOKEN_INDEXES])
                indexes.add(end)
                last_ngram = last_ngram_item[NGRAM]
                ngram = "%s %s" % (last_ngram, tokens[end])
            else:
                indexes = set(xrange(start, end + 1))
                ngram = " ".join(tokens[start:end + 1])
            ngram_item = {NGRAM: ngram, TOKEN_INDEXES: indexes}
            last_ngram_item = ngram_item
            local_ngrams.append(ngram_item)
        ngrams += local_ngrams
    return ngrams


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


LOWER_REGEX = re.compile(r"^[^A-Z]+$")
UPPER_REGEX = re.compile(r"^[^a-z]+$")
TITLE_REGEX = re.compile(r"^[A-Z][^A-Z]+$")
