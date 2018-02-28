from __future__ import unicode_literals

from snips_nlu.resources import get_stems
from snips_nlu.tokenization import tokenize_light


def stem(string, language):
    tokens = tokenize_light(string, language)
    stemmed_tokens = [_stem(token, language) for token in tokens]
    return ' '.join(stemmed_tokens)


def _stem(string, language):
    return get_stems(language).get(string, string)
