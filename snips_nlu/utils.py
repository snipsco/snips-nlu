from __future__ import unicode_literals

import errno
import functools
import numbers
import os

from builtins import object
from collections import Mapping, namedtuple
from threading import RLock

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

    - https://github.com/rust-lang/regex/blob/master/regex-syntax/src/
    lib.rs#L1685
    - https://github.com/rust-lang/regex/blob/master/regex-syntax/src/
    parser.rs#L1378
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


_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


@functools.wraps(functools.update_wrapper)
def update_wrapper(wrapper,
                   wrapped,
                   assigned=functools.WRAPPER_ASSIGNMENTS,
                   updated=functools.WRAPPER_UPDATES):
    """
    Patch two bugs in functools.update_wrapper.
    """
    # workaround for http://bugs.python.org/issue3445
    assigned = tuple(attr for attr in assigned if hasattr(wrapped, attr))
    wrapper = functools.update_wrapper(wrapper, wrapped, assigned, updated)
    # workaround for https://bugs.python.org/issue17482
    wrapper.__wrapped__ = wrapped
    return wrapper


class _HashedSeq(list):
    # pylint: disable=single-string-used-for-slots
    __slots__ = 'hashvalue'

    # pylint: disable=super-init-not-called
    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue

# pylint: disable=dangerous-default-value
def _make_key(args, kwds, typed,
              kwd_mark=(object(),),
              fasttypes=set([int, str, frozenset, type(None)]),
              sorted=sorted, tuple=tuple, type=type, len=len):
    'Make a cache key from optionally typed positional and keyword arguments'
    key = tuple(tuple(a) if isinstance(a, list) else a for a in args)
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            if isinstance(item[1], list):
                item = (item[0], tuple(item[1]))
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def lru_cache(maxsize=100, typed=False):
    """Least-recently-used cache decorator.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
     with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    """

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    def decorating_function(user_function):

        cache = dict()
        stats = [0, 0]  # make statistics updateable non-locally
        hits, misses = 0, 1  # names for the stats fields
        make_key = _make_key
        cache_get = cache.get  # bound method to lookup key or return None
        len_ = len  # localize the global len() function
        lock = RLock()  # because linkedlist updates aren't threadsafe
        root = []  # root of the circular doubly linked list
        root[:] = [root, root, None, None]  # initialize by pointing to self
        nonlocal_root = [root]  # make updateable non-locally
        prev_, next_, key_, result_ = 0, 1, 2, 3  # names for the link fields

        if maxsize == 0:

            def wrapper(*args, **kwds):
                # no caching, just do a statistics update after a successful
                # call
                result = user_function(*args, **kwds)
                stats[misses] += 1
                return result

        elif maxsize is None:

            def wrapper(*args, **kwds):
                # simple caching without ordering or size limit
                key = make_key(args, kwds, typed)
                # root used here as a unique not-found sentinel
                result = cache_get(key, root)
                if result is not root:
                    stats[hits] += 1
                    return result
                result = user_function(*args, **kwds)
                cache[key] = result
                stats[misses] += 1
                return result

        else:

            def wrapper(*args, **kwds):
                # size limited caching that tracks accesses by recency
                key = make_key(args, kwds, typed)
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # record recent use of the key by moving it to the
                        #  front of the list
                        root, = nonlocal_root
                        link_prev, link_next, key, result = link
                        link_prev[next_] = link_next
                        link_next[prev_] = link_prev
                        last = root[prev_]
                        last[next_] = root[prev_] = link
                        link[prev_] = last
                        link[next_] = root
                        stats[hits] += 1
                        return result
                result = user_function(*args, **kwds)
                with lock:
                    root, = nonlocal_root
                    if key in cache:
                        # getting here means that this same key was added to
                        # the cache while the lock was released.  since the
                        #  link update is already done, we need only return the
                        # computed result and update the count of misses.
                        pass
                    elif len_(cache) >= maxsize:
                        # use the old root to store the new key and result
                        oldroot = root
                        oldroot[key_] = key
                        oldroot[result_] = result
                        # empty the oldest link and make it the new root
                        root = nonlocal_root[0] = oldroot[next_]
                        oldkey = root[key_]
                        root[key_] = root[result_] = None
                        # now update the cache dictionary for the new links
                        del cache[oldkey]
                        cache[key] = oldroot
                    else:
                        # put result in a new link at the front of the list
                        last = root[prev_]
                        link = [last, root, key, result]
                        last[next_] = root[prev_] = cache[key] = link
                    stats[misses] += 1
                return result

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CacheInfo(stats[hits], stats[misses], maxsize,
                                  len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            with lock:
                cache.clear()
                root = nonlocal_root[0]
                root[:] = [root, root, None, None]
                stats[:] = [0, 0]

        wrapper.__wrapped__ = user_function
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)

    return decorating_function
