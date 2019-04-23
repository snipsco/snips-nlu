from __future__ import unicode_literals

import importlib
import json
import numbers
from builtins import bytes as newbytes, str as newstr
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import pkg_resources
from future.utils import text_type

from snips_nlu.constants import (
    END, START, RES_MATCH_RANGE, ENTITY_KIND, RES_VALUE)
from snips_nlu.exceptions import NotTrained, PersistingError

REGEX_PUNCT = {'\\', '.', '+', '*', '?', '(', ')', '|', '[', ']', '{', '}',
               '^', '$', '#', '&', '-', '~'}


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


def json_debug_string(dict_data):
    return json.dumps(dict_data, ensure_ascii=False, indent=2, sort_keys=True)


def json_string(json_object, indent=2, sort_keys=True):
    json_dump = json.dumps(json_object, indent=indent, sort_keys=sort_keys,
                           separators=(',', ': '))
    return unicode_string(json_dump)


def unicode_string(string):
    if isinstance(string, text_type):
        return string
    if isinstance(string, bytes):
        return string.decode("utf8")
    if isinstance(string, newstr):
        return text_type(string)
    if isinstance(string, newbytes):
        string = bytes(string).decode("utf8")

    raise TypeError("Cannot convert %s into unicode string" % type(string))


def check_persisted_path(func):
    @wraps(func)
    def func_wrapper(self, path, *args, **kwargs):
        path = Path(path)
        if path.exists():
            raise PersistingError(path)
        return func(self, path, *args, **kwargs)

    return func_wrapper


def fitted_required(func):
    @wraps(func)
    def func_wrapper(self, *args, **kwargs):
        if not self.fitted:
            raise NotTrained("%s must be fitted" % self.unit_name)
        return func(self, *args, **kwargs)

    return func_wrapper


def is_package(name):
    """Check if name maps to a package installed via pip.

    Args:
        name (str): Name of package

    Returns:
        bool: True if an installed packaged corresponds to this name, False
            otherwise.
    """
    name = name.lower().replace("-", "_")
    packages = pkg_resources.working_set.by_key.keys()
    for package in packages:
        if package.lower().replace("-", "_") == name:
            return True
    return False


def get_package_path(name):
    """Get the path to an installed package.

    Args:
        name (str): Package name

    Returns:
        class:`.Path`: Path to the installed package
    """
    name = name.lower().replace("-", "_")
    pkg = importlib.import_module(name)
    return Path(pkg.__file__).parent


def deduplicate_overlapping_items(items, overlap_fn, sort_key_fn):
    sorted_items = sorted(items, key=sort_key_fn)
    deduplicated_items = []
    for item in sorted_items:
        if not any(overlap_fn(item, dedup_item)
                   for dedup_item in deduplicated_items):
            deduplicated_items.append(item)
    return deduplicated_items


def replace_entities_with_placeholders(text, entities, placeholder_fn):
    if not entities:
        return dict(), text

    entities = deduplicate_overlapping_entities(entities)
    entities = sorted(
        entities, key=lambda e: e[RES_MATCH_RANGE][START])

    range_mapping = dict()
    processed_text = ""
    offset = 0
    current_ix = 0
    for ent in entities:
        ent_start = ent[RES_MATCH_RANGE][START]
        ent_end = ent[RES_MATCH_RANGE][END]
        rng_start = ent_start + offset

        processed_text += text[current_ix:ent_start]

        entity_length = ent_end - ent_start
        entity_place_holder = placeholder_fn(ent[ENTITY_KIND])

        offset += len(entity_place_holder) - entity_length

        processed_text += entity_place_holder
        rng_end = ent_end + offset
        new_range = (rng_start, rng_end)
        range_mapping[new_range] = ent[RES_MATCH_RANGE]
        current_ix = ent_end

    processed_text += text[current_ix:]
    return range_mapping, processed_text


def deduplicate_overlapping_entities(entities):
    def overlap(lhs_entity, rhs_entity):
        return ranges_overlap(lhs_entity[RES_MATCH_RANGE],
                              rhs_entity[RES_MATCH_RANGE])

    def sort_key_fn(entity):
        return -len(entity[RES_VALUE])

    deduplicated_entities = deduplicate_overlapping_items(
        entities, overlap, sort_key_fn)
    return sorted(deduplicated_entities,
                  key=lambda entity: entity[RES_MATCH_RANGE][START])
