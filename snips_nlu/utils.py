from __future__ import unicode_literals

import errno
import json
import numbers
import os

from builtins import object, str
from collections import OrderedDict, namedtuple, Mapping
from datetime import datetime

import numpy as np


from snips_nlu.constants import (INTENTS, UTTERANCES, DATA, SLOT_NAME, ENTITY,
                                 RESOURCES_PATH, END, START)

REGEX_PUNCT = {'\\', '.', '+', '*', '?', '(', ')', '|', '[', ']', '{', '}',
               '^', '$', '#', '&', '-', '~'}


class NotTrained(LookupError):
    """Exception used when a processing unit is used while not fitted"""
    pass


class ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


# pylint: enable=C0103


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
    return os.path.join(RESOURCES_PATH, language)


def mkdir_p(path):
    """Reproduces the 'mkdir -p shell' command

    See
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# pylint:disable=line-too-long
def regex_escape(s):
    """Escapes all regular expression meta characters in *s*

    The string returned may be safely used as a literal in a regular
    expression.

    This function is more precise than :func:`re.escape`, the latter escapes
    all non-alphanumeric characters which can cause cross-platform
    compatibility issues.

    References:

    - https://github.com/rust-lang/regex/blob/master/regex-syntax/src/lib.rs#L1685
    - https://github.com/rust-lang/regex/blob/master/regex-syntax/src/parser.rs#L1378
    """
    escaped_string = ""
    for c in s:
        if c in REGEX_PUNCT:
            escaped_string += "\\"
        escaped_string += c
    return escaped_string


# pylint:enable=line-too-long


def check_random_state(seed):
    """Turn seed into a :class:`numpy.random.RandomState` instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    # pylint: disable=W0212
    # pylint: disable=c-extension-no-member
    if seed is None or seed is np.random:
        return np.random.mtrand._rand  # pylint: disable=c-extension-no-member
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def get_slot_name_mapping(dataset, intent):
    """Returns a dict which maps slot names to entities for the provided intent
    """
    slot_name_mapping = dict()
    for utterance in dataset[INTENTS][intent][UTTERANCES]:
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name_mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def get_slot_name_mappings(dataset):
    """Returns a dict which maps intents to their slot name mapping"""
    return {intent: get_slot_name_mapping(dataset, intent)
            for intent in dataset[INTENTS]}


def ranges_overlap(lhs_range, rhs_range):
    if isinstance(lhs_range, dict) and isinstance(rhs_range, dict):
        return lhs_range[END] > rhs_range[START] \
               and lhs_range[START] < rhs_range[END]
    elif isinstance(lhs_range, (tuple, list)) \
            and isinstance(rhs_range, (tuple, list)):
        return lhs_range[1] > rhs_range[0] and lhs_range[0] < rhs_range[1]
    else:
        raise TypeError("Cannot check overlap on objects of type: %s and %s"
                        % (type(lhs_range), type(rhs_range)))


def elapsed_since(time):
    return datetime.now() - time


class DifferedLoggingMessage(object):

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.fn(*self.args, **self.kwargs))


def json_debug_string(dict_data):
    return json.dumps(dict_data, ensure_ascii=False, indent=2, sort_keys=True)


def log_elapsed_time(logger, level, output_msg=None):
    if output_msg is None:
        output_msg = "Elapsed time ->:\n{elapsed_time}"

    def get_wrapper(fn):
        def wrapped(*args, **kwargs):
            start = datetime.now()
            msg_fmt = dict()
            res = fn(*args, **kwargs)
            if "elapsed_time" in output_msg:
                msg_fmt["elapsed_time"] = datetime.now() - start
            logger.log(level, output_msg.format(**msg_fmt))
            return res

        return wrapped

    return get_wrapper


def log_result(logger, level, output_msg=None):
    if output_msg is None:
        output_msg = "Result ->:\n{result}"

    def get_wrapper(fn):
        def wrapped(*args, **kwargs):
            msg_fmt = dict()
            res = fn(*args, **kwargs)
            if "result" in output_msg:
                try:
                    res_debug_string = json_debug_string(res)
                except TypeError:
                    res_debug_string = str(res)
                msg_fmt["result"] = res_debug_string
            logger.log(level, output_msg.format(**msg_fmt))
            return res

        return wrapped

    return get_wrapper
