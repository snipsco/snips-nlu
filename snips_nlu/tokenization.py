# coding=utf-8
from __future__ import unicode_literals

from jieba import Tokenizer
from nlu_utils import (tokenize as _tokenize,
                       tokenize_light as _tokenize_light,
                       normalize)

from snips_nlu.languages import Language

JIEBA_TOKENIZER = Tokenizer()


def initialize_jieba_tokenizer():
    JIEBA_TOKENIZER.check_initialized()


class Token(object):
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
        else:
            self._normalized_value = normalize(self.value)
            return self._normalized_value

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return (self.value == other.value
                and self.start == other.start
                and self.end == other.end
                and self.stem == other.stem)

    def __ne__(self, other):
        return not self.__eq__(other)


def tokenize(string, language):
    if language == Language.ZH:
        if not isinstance(string, unicode):
            string = string.decode("utf8")
        jiebla_tokens = JIEBA_TOKENIZER.tokenize(string)
        tokens = [Token(value=value, start=start, end=end)
                  for value, start, end in jiebla_tokens
                  if value.strip() not in language.punctuation]
    else:
        tokens = [Token(value=token["value"],
                        start=token["char_range"]["start"],
                        end=token["char_range"]["end"])
                  for token in _tokenize(string)]
    return tokens


def tokenize_light(string, language):
    if language == Language.ZH:
        tokenized_string = [t.value for t in tokenize(string, language)]
    else:
        tokenized_string = _tokenize_light(string)
    return tokenized_string
