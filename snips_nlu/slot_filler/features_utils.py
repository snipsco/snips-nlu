from snips_nlu.constants import NGRAM, TOKEN_INDEXES


def get_all_ngrams(tokens, max_ngram_size):
    ngrams = []
    for start in range(len(tokens)):
        local_ngrams = []
        last_ngram_item = None
        max_end = min(len(tokens), start + max_ngram_size)
        for end in range(start, max_end):
            if last_ngram_item is not None:
                indexes = last_ngram_item[TOKEN_INDEXES] + [end]
                last_ngram = last_ngram_item[NGRAM]
                ngram = "%s %s" % (last_ngram, tokens[end])
            else:
                indexes = [start]
                ngram = tokens[start]
            ngram_item = {NGRAM: ngram, TOKEN_INDEXES: indexes}
            last_ngram_item = ngram_item
            local_ngrams.append(ngram_item)
        ngrams += local_ngrams
    return ngrams
