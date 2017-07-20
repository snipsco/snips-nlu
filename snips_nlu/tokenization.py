# coding=utf-8
from __future__ import unicode_literals

from nlu_utils import tokenize as _tokenize, tokenize_light

from snips_nlu.utils import namedtuple_with_defaults

Token = namedtuple_with_defaults('Token', 'value start end stem', {
    'stem': None})


def tokenize(string):
    return [
        Token(
            value=token["value"],
            start=token["char_range"]["start"],
            end=token["char_range"]["end"])
        for token in _tokenize(string)]
