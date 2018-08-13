from __future__ import unicode_literals

from builtins import object, str

from snips_nlu_ontology import (
    BuiltinEntityParser as _BuiltinEntityParser, get_all_builtin_entities,
    get_supported_entities)

from snips_nlu.utils import LimitedSizeDict


class BuiltinEntityParser(object):
    def __init__(self, language):
        self.language = language
        self.parser = _BuiltinEntityParser(language)
        self.supported_entities = get_supported_entities(language)
        self._cache = LimitedSizeDict(size_limit=1000)

    def parse(self, text, scope=None, use_cache=True):
        if use_cache is not True:
            text_lower = text.lower()  # Rustling only works with lowercase
            self.parser.parse(text_lower, scope)
        cache_key = (text, scope)
        parser_result = self._cache.get(cache_key)
        if parser_result is None:
            text_lower = text.lower()  # Rustling only works with lowercase
            parser_result = self.parser.parse(text_lower, scope)
            self._cache[cache_key] = parser_result
        return parser_result

    def supports_entity(self, entity):
        return entity in self.supported_entities


_RUSTLING_PARSERS = dict()


def get_builtin_entity_parser(language):
    global _RUSTLING_PARSERS
    if language not in _RUSTLING_PARSERS:
        _RUSTLING_PARSERS[language] = BuiltinEntityParser(language)
    return _RUSTLING_PARSERS[language]


def get_builtin_entities(text, language, scope=None, use_cache=True):
    parser = get_builtin_entity_parser(language)
    return parser.parse(text, scope=scope, use_cache=use_cache)


def is_builtin_entity(entity_label):
    return entity_label in get_all_builtin_entities()
