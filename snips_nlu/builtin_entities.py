from __future__ import unicode_literals

import six
from builtins import object
from snips_nlu_ontology import (
    get_all_builtin_entities, BuiltinEntityParser as _BuiltinEntityParser,
    get_supported_entities)

if six.PY2:
    from backports.functools_lru_cache import lru_cache
else:
    from functools import lru_cache


class BuiltinEntityParser(object):

    def __init__(self, language, cache_size=1000):
        self.language = language
        self.cache_size = cache_size
        self.parser = _BuiltinEntityParser(language)
        self.supported_entities = get_supported_entities(language)
        self.parse = lru_cache(cache_size)(self.parse)

    def parse(self, text, scope=None):
        text = text.lower()  # Rustling only works with lowercase
        return self.parser.parse(text, scope)

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
