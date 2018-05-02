from __future__ import unicode_literals

import re
import string

from num2words import num2words

_PUNCTUATION_REGEXES = dict()
_NUM2WORDS_SUPPORT = dict()


# pylint:disable=unused-argument
def get_default_sep(language):
    return " "


# pylint:enable=unused-argument

# pylint:disable=unused-argument
def get_punctuation(language):
    return string.punctuation


# pylint:enable=unused-argument


def get_punctuation_regex(language):
    global _PUNCTUATION_REGEXES
    if language not in _PUNCTUATION_REGEXES:
        pattern = r"|".join(re.escape(p) for p in get_punctuation(language))
        _PUNCTUATION_REGEXES[language] = re.compile(pattern)
    return _PUNCTUATION_REGEXES[language]


def supports_num2words(language):
    global _NUM2WORDS_SUPPORT

    if language not in _NUM2WORDS_SUPPORT:
        try:
            num2words(0, lang=language)
            _NUM2WORDS_SUPPORT[language] = True
        except NotImplementedError:
            _NUM2WORDS_SUPPORT[language] = False
    return _NUM2WORDS_SUPPORT[language]
