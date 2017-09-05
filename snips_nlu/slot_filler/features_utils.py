from __future__ import unicode_literals

from nlu_utils import compute_all_ngrams

from snips_nlu.utils import LimitedSizeDict

_NGRAMS_CACHE = LimitedSizeDict(size_limit=1000)


def get_all_ngrams(tokens):
    key = "<||>".join(tokens)
    if key not in _NGRAMS_CACHE:
        ngrams = compute_all_ngrams(tokens, len(tokens))
        _NGRAMS_CACHE[key] = ngrams
    return _NGRAMS_CACHE[key]


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
