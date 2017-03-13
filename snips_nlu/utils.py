from collections import OrderedDict

import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


def sequence_equal(seq, other_seq):
    return len(seq) == len(other_seq) and sorted(seq) == sorted(other_seq)


def merge_two_dicts(x, y, shallow_copy=True):

    """Given two dicts, merge them into a new dict.
    :param x: first dict
    :param y: second dict
    :param shallow_copy: if False, `x` will be updated with y and returned.
    Otherwise a shallow copy of `x` will be created (default).
    """
    z = x.copy() if shallow_copy else x
    z.update(y)
    return z


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        if "size_limit" not in kwds:
            raise ValueError("'size_limit' must be passed as a keyword "
                             "argument")
        self.size_limit = kwds.pop("size_limit")
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        if len(args) == 1 and len(args[0]) + len(kwds) > self.size_limit:
            raise ValueError("Tried to initialize LimitedSizedDict with more "
                             "value than permitted with 'limit_size'")
        super(LimitedSizeDict, self).__init__(*args, **kwds)

    def __setitem__(self, key, value, dict_setitem=OrderedDict.__setitem__):
        dict_setitem(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)

    def __eq__(self, other):
        if self.size_limit != other.size_limit:
            return False
        return super(LimitedSizeDict, self).__eq__(other)
