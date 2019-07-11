# coding=utf-8
from __future__ import unicode_literals

from builtins import object

from snips_nlu.resources import get_stems


def stem(string, language, resources):
    from snips_nlu_utils import normalize

    normalized_string = normalize(string)
    tokens = tokenize_light(normalized_string, language)
    stemmed_tokens = [_stem(token, resources) for token in tokens]
    return " ".join(stemmed_tokens)


def stem_token(token, resources):
    from snips_nlu_utils import normalize

    if token.stemmed_value:
        return token.stemmed_value
    if not token.normalized_value:
        token.normalized_value = normalize(token.value)
    token.stemmed_value = _stem(token.normalized_value, resources)
    return token.stemmed_value


def normalize_token(token):
    from snips_nlu_utils import normalize

    if token.normalized_value:
        return token.normalized_value
    token.normalized_value = normalize(token.value)
    return token.normalized_value


def _stem(string, resources):
    return get_stems(resources).get(string, string)


class Token(object):
    """Token object which is output by the tokenization

    Attributes:
        value (str): Tokenized string
        start (int): Start position of the token within the sentence
        end (int): End position of the token within the sentence
        normalized_value (str): Normalized value of the tokenized string
        stemmed_value (str): Stemmed value of the tokenized string
    """

    def __init__(self, value, start, end, normalized_value=None,
                 stemmed_value=None):
        self.value = value
        self.start = start
        self.end = end
        self.normalized_value = normalized_value
        self.stemmed_value = stemmed_value

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
    from snips_nlu_utils import tokenize as _tokenize

    tokens = [Token(value=token["value"],
                    start=token["char_range"]["start"],
                    end=token["char_range"]["end"])
              for token in _tokenize(string, language)]
    return tokens


def tokenize_light(string, language):
    """Same behavior as :func:`tokenize` but returns tokenized strings instead
        of :class:`Token` objects"""
    from snips_nlu_utils import tokenize_light as _tokenize_light

    tokenized_string = _tokenize_light(string, language)
    return tokenized_string
