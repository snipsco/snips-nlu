# coding=utf-8
from __future__ import unicode_literals

from builtins import object

from snips_nlu_utils import (tokenize as _tokenize,
                             tokenize_light as _tokenize_light,
                             normalize)


class Token(object):
    """Token object which is output by the tokenization

    Attributes:
        value (str): Tokenized string
        normalized (str): Normalized value of the tokenized string
        stem (str, optional): Stemmed value, if defined, of the tokenized
            string
        start (int): Start position of the token within the sentence
        end (int): End position of the token within the sentence
    """

    def __init__(self, value, start, end, normalized=None, stem=None):
        self.value = value
        self.start = start
        self.end = end
        self._normalized_value = normalized
        self.stem = stem

    @property
    def normalized_value(self):
        if self._normalized_value is not None:
            return self._normalized_value
        self._normalized_value = normalize(self.value)
        return self._normalized_value

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (self.value == other.value
                and self.start == other.start
                and self.end == other.end
                and self.stem == other.stem)

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
