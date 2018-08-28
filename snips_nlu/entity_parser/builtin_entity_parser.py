from __future__ import unicode_literals

import json

from snips_nlu_ontology import (
    BuiltinEntityParser as _BuiltinEntityParser, get_all_builtin_entities,
    get_all_gazetteer_entities, get_all_grammar_entities,
    get_supported_gazetteer_entities)

from snips_nlu.constants import DATA_PATH, ENTITIES, LANGUAGE
from snips_nlu.entity_parser.entity_parser import EntityParser, _get_caching_key


class BuiltinEntityParser(EntityParser):
    def __init__(self, language, entity_configurations):
        if entity_configurations is None:
            entity_configurations = []
        self.language = language
        self.entity_configurations = entity_configurations
        self._parser = _BuiltinEntityParser(language, entity_configurations)

    @property
    def parser(self):
        return self._parser


_BUILTIN_ENTITY_PARSERS = dict()


def get_builtin_entity_parser(dataset):
    language = dataset[LANGUAGE]
    gazetteer_entities = [entity for entity in dataset[ENTITIES]
                          if is_gazetteer_entity(entity)]
    return get_builtin_entity_parser_from_scope(language, gazetteer_entities)


def get_builtin_entity_parser_from_scope(language, gazetteer_entity_scope):
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


def is_builtin_entity(entity_label):
    return entity_label in get_all_builtin_entities()


def is_gazetteer_entity(entity_label):
    return entity_label in get_all_gazetteer_entities()


def is_grammar_entity(entity_label):
    return entity_label in get_all_grammar_entities()


def find_gazetteer_entity_data_path(language, entity_name):
    for directory in DATA_PATH.iterdir():
        if not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open(encoding="utf8") as f:
            metadata = json.load(f)
        if metadata.get("entity_name") == entity_name \
                and metadata.get("language") == language:
            return directory / metadata["data_directory"]
    raise FileNotFoundError(
        "No data found for the '{e}' builtin entity in language '{lang}'. "
        "You must download the corresponding resources by running "
        "'python -m snips_nlu download-entity {e} {lang}' before you can use "
        "this builtin entity.".format(e=entity_name, lang=language))


def _get_gazetteer_entity_configurations(language, gazetteer_entity_scope):
    return [{
        "builtin_entity_name": entity_name,
        "resource_path": str(find_gazetteer_entity_data_path(
            language, entity_name))
    } for entity_name in gazetteer_entity_scope]
