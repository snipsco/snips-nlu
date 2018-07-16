# coding=utf-8
from __future__ import unicode_literals

from builtins import object

from snips_nlu_utils import (
    normalize, tokenize as _tokenize, tokenize_light as _tokenize_light)

from snips_nlu.resources import get_stems


def stem(string, language):
    tokens = tokenize_light(string, language)
    stemmed_tokens = [_stem(token, language) for token in tokens]
    return " ".join(stemmed_tokens)


def _stem(string, language):
    return get_stems(language).get(string, string)


class Token(object):
    """Token object which is output by the tokenization

    Attributes:
        value (str): Tokenized string
        normalized (str): Normalized value of the tokenized string
        start (int): Start position of the token within the sentence
        end (int): End position of the token within the sentence
    """

    def __init__(self, value, start, end):
        self.value = value
        self.start = start
        self.end = end
        self._normalized_value = None
        self._stem = None

    def get_normalized_value(self):
        if self._normalized_value is not None:
            return self._normalized_value
        self._normalized_value = normalize(self.value)
        return self._normalized_value

    def get_stem(self, language):
        if self._stem is None:
            self._stem = stem(normalize(self.value), language)
        return self._stem

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (self.value == other.value
                and self.start == other.start
                and self.end == other.end)

    def __ne__(self, other):
        return not self.__eq__(other)


def tokenize(string, language):
    """Tokenizes the input

    Args:
        string (str): Input to tokenize
        language (str): Language to use during tokenization

    Returns:
        list of :class:`.Token`: The list of tokenized values
    """
    tokens = [Token(value=token["value"],
                    start=token["char_range"]["start"],
                    end=token["char_range"]["end"])
              for token in _tokenize(string, language)]
    return tokens


def tokenize_light(string, language):
    """Same behavior as :func:`tokenize` but returns tokenized strings instead
        of :class:`Token` objects"""
    tokenized_string = _tokenize_light(string, language)
    return tokenized_string
