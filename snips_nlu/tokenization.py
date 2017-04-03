import re
from collections import namedtuple

from snips_nlu.utils import namedtuple_with_defaults

Token = namedtuple_with_defaults('Token', 'value start end stem', {
    'stem': None})

TOKEN_REGEX = re.compile(r"\w+", re.UNICODE)


def tokenize(string):
    return [Token(m.group(), m.start(), m.end())
            for m in TOKEN_REGEX.finditer(string)]
