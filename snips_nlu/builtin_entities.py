from __future__ import unicode_literals

import json
from builtins import object, str

from snips_nlu_ontology import (
    BuiltinEntityParser as _BuiltinEntityParser, get_all_builtin_entities,
    get_all_gazetteer_entities, get_all_grammar_entities,
    get_supported_gazetteer_entities)

from snips_nlu.constants import DATA_PATH, LANGUAGE
from snips_nlu.utils import LimitedSizeDict


class BuiltinEntityParser(object):
    def __init__(self, language, gazetteer_entity_configurations):
        if gazetteer_entity_configurations is None:
            gazetteer_entity_configurations = []
        self.language = language
        self.gazetteer_entity_configurations = gazetteer_entity_configurations
        self.parser = _BuiltinEntityParser(
            language, gazetteer_entity_configurations)
        self._cache = LimitedSizeDict(size_limit=1000)

    def parse(self, text, scope=None, use_cache=True):
        text = text.lower()
        if not use_cache:
            return self.parser.parse(text, scope)
        cache_key = (text, str(scope))
        if cache_key not in self._cache:
            parser_result = self.parser.parse(text, scope)
            self._cache[cache_key] = parser_result
        return self._cache[cache_key]


_BUILTIN_ENTITY_PARSERS = dict()


def get_builtin_entity_parser(dataset):
    language = dataset[LANGUAGE]
    gazetteer_entities = [entity for entity in dataset["entities"]
                          if is_gazetteer_entity(entity)]
    return _get_builtin_entity_parser(language, gazetteer_entities)


def find_gazetteer_entity_data_path(language, entity_name):
    for directory in DATA_PATH.iterdir():
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open(encoding="utf8") as f:
            metadata = json.load(f)
        if metadata.get("entity_name") == entity_name \
                and metadata.get("language") == language:
            return directory / metadata["data_directory"]
    raise FileNotFoundError("No gazetteer entity data found for '%s' in "
                            "language '%s'" % (entity_name, language))


def is_builtin_entity(entity_label):
    return entity_label in get_all_builtin_entities()


def is_gazetteer_entity(entity_label):
    return entity_label in get_all_gazetteer_entities()


def is_grammar_entity(entity_label):
    return entity_label in get_all_grammar_entities()


def _get_builtin_entity_parser(language, gazetteer_entity_scope):
    global _BUILTIN_ENTITY_PARSERS
    caching_key = _get_caching_key(language, gazetteer_entity_scope)
    if caching_key not in _BUILTIN_ENTITY_PARSERS:
        for entity in gazetteer_entity_scope:
            if entity not in get_supported_gazetteer_entities(language):
                raise ValueError("Gazetteer entity '%s' is not supported in "
                                 "language '%s'" % (entity, language))
        configurations = _get_gazetteer_entity_configurations(
            language, gazetteer_entity_scope)
        _BUILTIN_ENTITY_PARSERS[caching_key] = BuiltinEntityParser(
            language, configurations)
    return _BUILTIN_ENTITY_PARSERS[caching_key]


def _get_gazetteer_entity_configurations(language, gazetteer_entity_scope):
    return [{
        "builtin_entity_name": entity_name,
        "resource_path": str(find_gazetteer_entity_data_path(
            language, entity_name)),
        "parser_threshold": 0.6
    } for entity_name in gazetteer_entity_scope]


def _get_caching_key(language, gazetteer_entity_scope):
    tuple_key = (language,)
    tuple_key += tuple(entity for entity in sorted(gazetteer_entity_scope))
    return tuple_key
