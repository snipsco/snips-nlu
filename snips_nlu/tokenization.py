# coding=utf-8
from __future__ import unicode_literals

import re

from snips_nlu.utils import namedtuple_with_defaults

Token = namedtuple_with_defaults('Token', 'value start end stem', {
    'stem': None})

CURRENCIES = "$؋ƒ៛¥₡₱£€¢﷼₪₩₭₨₮₦₽฿₴₫"
WORD_REGEX = re.compile(r"\w+", re.UNICODE)
SYMBOL_REGEX = re.compile(r"[?!&%{}]+".format(CURRENCIES), re.UNICODE)
REGEXES_LIST = [WORD_REGEX, SYMBOL_REGEX]


def tokenize(string):
    return _tokenize(string, REGEXES_LIST)


def _tokenize(string, regexes):
    non_overlapping_tokens = []
    for regex in regexes:
        tokens = [Token(m.group(), m.start(), m.end())
                  for m in regex.finditer(string)]
        tokens = filter(
            lambda s: all(s.start >= t.end or s.end <= t.start for t in
                          non_overlapping_tokens),
            tokens)
        non_overlapping_tokens += tokens
    return sorted(non_overlapping_tokens, key=lambda t: t.start)


def tokenize_light(string):
    return [token[0] for token in tokenize(string)]
