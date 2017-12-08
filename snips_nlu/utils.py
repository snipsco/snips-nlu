from __future__ import unicode_literals

import errno
import numbers
import os
from collections import OrderedDict, namedtuple, Mapping

import numpy as np

from snips_nlu.constants import INTENTS, UTTERANCES, DATA, SLOT_NAME, ENTITY, \
    RESOURCES_PATH

REGEX_PUNCT = {'\\', '.', '+', '*', '?', '(', ')', '|', '[', ']', '{', '}',
               '^', '$', '#', '&', '-', '~'}


# pylint: disable=C0103
class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    # pylint: disable=W0622
    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)
    # pylint: enable=W0622


class classproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


# pylint: enable=C0103

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


def type_error(expected_type, found_type):
    return TypeError("Expected %s but found: %s" % (expected_type, found_type))


def validate_type(obj, expected_type):
    if not isinstance(obj, expected_type):
        raise type_error(expected_type=expected_type, found_type=type(obj))


def missing_key_error(key, object_label=None):
    if object_label is None:
        return KeyError("Missing key: '%s'" % key)
    return KeyError("Expected %s to have key: '%s'" % (object_label, key))


def validate_key(obj, key, object_label=None):
    if key not in obj:
        raise missing_key_error(key, object_label)


def validate_keys(obj, keys, object_label=None):
    for key in keys:
        validate_key(obj, key, object_label)


def validate_range(rng):
    if not isinstance(rng, (list, tuple)) or len(rng) != 2 or rng[0] > rng[1]:
        raise ValueError("range must be a length 2 list or tuple and must be "
                         "valid")


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


class UnupdatableDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise KeyError("Can't update key '%s'" % key)
        super(UnupdatableDict, self).__setitem__(key, value)


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = namedtuple(typename, field_names)  # pylint: disable=C0103
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def get_resources_path(language):
    return os.path.join(RESOURCES_PATH, language.iso_code)


def ensure_string(string_or_unicode, encoding="utf8"):
    if isinstance(string_or_unicode, str):
        return string_or_unicode
    elif isinstance(string_or_unicode, unicode):
        return string_or_unicode.encode(encoding)
    else:
        raise TypeError("Expected str or unicode, found %s"
                        % type(string_or_unicode))


def mkdir_p(path):
    """
    Reproduces the mkdir -p shell command, see
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def regex_escape(s):
    """
    Escapes all regular expression meta characters in `text`.

    The string returned may be safely used as a literal in a regular
     expression.

    This function is more precise than `re.escape`, the latter escapes
    all non-alphanumeric characters which can cause cross-platform
    compatibility issues.

    References:
        https://github.com/rust-lang/regex/blob/master/regex-syntax/src/lib.rs#L1685
        https://github.com/rust-lang/regex/blob/master/regex-syntax/src/parser.rs#L1378
    """
    escaped_string = ""
    for c in s:
        if c in REGEX_PUNCT:
            escaped_string += "\\"
        escaped_string += c
    return escaped_string


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    # pylint: disable=W0212
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def get_slot_name_mapping(dataset):
    """
    Returns a dict which maps slot names to entities
    """
    slot_name_mapping = dict()
    for intent_name, intent in dataset[INTENTS].iteritems():
        mapping = dict()
        slot_name_mapping[intent_name] = mapping
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping
