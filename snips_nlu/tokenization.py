import re
from collections import namedtuple

Token = namedtuple('Token', 'value start end')

TOKEN_REGEX = re.compile(r"\w+", re.UNICODE)


def tokenize(string):
    return [Token(m.group(), m.start(), m.end())
            for m in TOKEN_REGEX.finditer(string)]
