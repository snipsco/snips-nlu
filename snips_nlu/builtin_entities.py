from __future__ import unicode_literals

import re

import six
from builtins import object
from snips_nlu_ontology import (
    get_all_builtin_entities, BuiltinEntityParser as _BuiltinEntityParser,
    get_supported_entities)

if six.PY2:
    from backports.functools_lru_cache import lru_cache
else:
    from functools import lru_cache

NON_SPACE_REGEX = re.compile("[^\s]+")


class BuiltinEntityParser(object):
    non_space_separated_languages = {"ja", "zh"}

    def __init__(self, language, cache_size=1000):
        self.language = language
        self.cache_size = cache_size
        self.parser = _BuiltinEntityParser(language)
        self.supported_entities = get_supported_entities(language)
        self.parse = lru_cache(cache_size)(self.parse)

    def parse(self, text, scope=None):
        """Extract builtin entities from a text
        Args:
            text (str): text to parse
            scope (tuple, optional): tuple of the name of the builtin entities
             to extract

        Returns:
            list: a list of builtin entities
        """
        text = text.lower()  # Rustling only works with lowercase
        if self.language not in self.non_space_separated_languages:
            return self.parser.parse(text, scope)

        non_space_ranges = [(m.start(), m.end())
                            for m in NON_SPACE_REGEX.finditer(text)]
        if not non_space_ranges:
            return []

        new_ranges = [(0, non_space_ranges[0][1] - non_space_ranges[0][0])]
        for r in non_space_ranges[1:]:
            new_start = new_ranges[-1][1]
            new_end = new_start + r[1] - r[0]
            new_ranges.append((new_start, new_end))

        match_end_to_index = {r[1]: i for i, r in enumerate(new_ranges)}

        joined_text = "".join(text[start:end]
                              for start, end in non_space_ranges)
        result = []
        for res in self.parser.parse(joined_text, scope=scope):
            start = res["range"]["start"]
            end = res["range"]["end"]
            if start == 0:
                start_ix = 0
            elif start in match_end_to_index:
                start_ix = match_end_to_index[start] + 1
            else:  # match does not correspond to token
                continue

            if end in match_end_to_index:
                end_ix = match_end_to_index[end]
            else:  # match does not correspond to token
                continue

            initial_start = non_space_ranges[start_ix][0]
            initial_end = non_space_ranges[end_ix][1]
            res["range"]["start"] = initial_start
            res["range"]["end"] = initial_end
            res["value"] = text[initial_start:initial_end]
            result.append(res)

        return result

    def supports_entity(self, entity):
        return entity in self.supported_entities


_RUSTLING_PARSERS = dict()


def get_builtin_entity_parser(language):
    global _RUSTLING_PARSERS
    if language not in _RUSTLING_PARSERS:
        _RUSTLING_PARSERS[language] = BuiltinEntityParser(language)
    return _RUSTLING_PARSERS[language]


def get_builtin_entities(text, language, scope=None):
    parser = get_builtin_entity_parser(language)
    return parser.parse(text, scope=scope)


def is_builtin_entity(entity_label):
    return entity_label in get_all_builtin_entities()
