from __future__ import unicode_literals

TOKEN_NAME = "token"


class Feature(object):
    def __init__(self, base_name, func, offset=0, drop_out=0):
        if base_name == TOKEN_NAME:
            raise ValueError("'%s' name is reserved" % TOKEN_NAME)
        self.offset = offset
        self._name = None
        self._base_name = None
        self.base_name = base_name
        self.function = func
        self.drop_out = drop_out

    @property
    def name(self):
        return self._name

    @property
    def base_name(self):
        return self._base_name

    @base_name.setter
    def base_name(self, value):
        self._name = _offset_name(value, self.offset)
        self._base_name = _offset_name(value, 0)

    def compute(self, token_index, cache):
        if not 0 <= (token_index + self.offset) < len(cache):
            return None

        if self.base_name in cache[token_index + self.offset]:
            return cache[token_index + self.offset][self.base_name]

        tokens = [c["token"] for c in cache]
        value = self.function(tokens, token_index + self.offset)
        cache[token_index + self.offset][self.base_name] = value
        return value


def _offset_name(name, offset):
    if offset > 0:
        return "%s[+%s]" % (name, offset)
    if offset < 0:
        return "%s[%s]" % (name, offset)
    return name
