from copy import copy

from snips_nlu.constants import NGRAM, TOKEN_INDEXES


def get_all_ngrams(tokens, max_ngram_size=None, keep_only_index=None):
    if max_ngram_size is None:
        max_ngram_size = len(tokens)
    max_start = len(tokens) - 1
    min_end = 0
    if keep_only_index is not None:
        max_start = keep_only_index
        min_end = keep_only_index

    ngrams = []
    for start in range(max_start + 1):
        local_ngrams = []
        last_ngram_item = None
        _min_end = max(start, min_end)
        _max_end = min(len(tokens), _min_end + max_ngram_size)
        for end in range(_min_end, _max_end):
            if last_ngram_item is not None:
                indexes = copy(last_ngram_item[TOKEN_INDEXES])
                indexes.add(end)
                last_ngram = last_ngram_item[NGRAM]
                ngram = "%s %s" % (last_ngram, tokens[end])
            else:
                indexes = set(range(start, end + 1))
                ngram = " ".join(tokens[start:end + 1])
            ngram_item = {NGRAM: ngram, TOKEN_INDEXES: indexes}
            last_ngram_item = ngram_item
            local_ngrams.append(ngram_item)
        ngrams += local_ngrams
    return ngrams
