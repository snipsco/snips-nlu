from __future__ import unicode_literals

from builtins import object

TOKEN_NAME = "token"


class Feature(object):
    """CRF Feature which is used by :class:`.CRFSlotFiller`

    Attributes:
        base_name (str): Feature name (e.g. 'is_digit', 'is_first' etc)
        func (function): The actual feature function for example:

            def is_first(tokens, token_index):
                return "1" if token_index == 0 else None

        offset (int, optional): Token offset to consider when computing
            the feature (e.g -1 for computing the feature on the previous word)
        drop_out (float, optional): Drop out to use when computing the
            feature during training

    Note:
        The easiest way to add additional features to the existing ones is
        to create a :class:`.CRFFeatureFactory`
    """

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

    def compute(self, token_index, cache, tokens, _initial_text):
        cache_index = token_index + self.offset
        if not 0 <= cache_index < len(cache):
            return None
        cache_dict = cache[cache_index]
        base_name = self.base_name
        value = cache_dict.get(base_name)
        if value is None:
            value = self.function(tokens, cache_index, _initial_text)
            cache_dict[base_name] = value
        return value


def _offset_name(name, offset):
    if offset > 0:
        return "%s[+%s]" % (name, offset)
    if offset < 0:
        return "%s[%s]" % (name, offset)
    return name
