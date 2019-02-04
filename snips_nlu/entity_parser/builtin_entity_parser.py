from __future__ import unicode_literals

import json
import shutil

from future.builtins import str
from snips_nlu_parsers import (
    BuiltinEntityParser as _BuiltinEntityParser, get_all_builtin_entities,
    get_all_gazetteer_entities, get_all_grammar_entities,
    get_builtin_entity_shortname, get_supported_gazetteer_entities)

from snips_nlu.common.io_utils import temp_dir
from snips_nlu.common.utils import json_string
from snips_nlu.constants import DATA_PATH, ENTITIES, LANGUAGE
from snips_nlu.entity_parser.entity_parser import EntityParser
from snips_nlu.result import parsed_entity

_BUILTIN_ENTITY_PARSERS = dict()

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class BuiltinEntityParser(EntityParser):
    def __init__(self, parser):
        super(BuiltinEntityParser, self).__init__()
        self._parser = parser

    def _parse(self, text, scope=None):
        entities = self._parser.parse(text.lower(), scope=scope)
        result = []
        for entity in entities:
            ent = parsed_entity(
                entity_kind=entity["entity_kind"],
                entity_value=entity["value"],
                entity_resolved_value=entity["entity"],
                entity_range=entity["range"]
            )
            result.append(ent)
        return result

    def persist(self, path):
        self._parser.persist(path)

    @classmethod
    def from_path(cls, path):
        parser = _BuiltinEntityParser.from_path(path)
        return cls(parser)

    @classmethod
    def build(cls, dataset=None, language=None, gazetteer_entity_scope=None):
        global _BUILTIN_ENTITY_PARSERS

        if dataset is not None:
            language = dataset[LANGUAGE]
            gazetteer_entity_scope = [entity for entity in dataset[ENTITIES]
                                      if is_gazetteer_entity(entity)]

        if language is None:
            raise ValueError("Either a dataset or a language must be provided "
                             "in order to build a BuiltinEntityParser")

        if gazetteer_entity_scope is None:
            gazetteer_entity_scope = []
        caching_key = _get_caching_key(language, gazetteer_entity_scope)
        if caching_key not in _BUILTIN_ENTITY_PARSERS:
            for entity in gazetteer_entity_scope:
                if entity not in get_supported_gazetteer_entities(language):
                    raise ValueError(
                        "Gazetteer entity '%s' is not supported in "
                        "language '%s'" % (entity, language))
            _BUILTIN_ENTITY_PARSERS[caching_key] = _build_builtin_parser(
                language, gazetteer_entity_scope)
        return _BUILTIN_ENTITY_PARSERS[caching_key]


def _build_builtin_parser(language, gazetteer_entities):
    with temp_dir() as serialization_dir:
        gazetteer_entity_parser = None
        if gazetteer_entities:
            gazetteer_entity_parser = _build_gazetteer_parser(
                serialization_dir, gazetteer_entities, language)

        metadata = {
            "language": language.upper(),
            "gazetteer_parser": gazetteer_entity_parser
        }
        metadata_path = serialization_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(json_string(metadata))
        parser = _BuiltinEntityParser.from_path(serialization_dir)
        return BuiltinEntityParser(parser)


def _build_gazetteer_parser(target_dir, gazetteer_entities, language):
    gazetteer_parser_name = "gazetteer_entity_parser"
    gazetteer_parser_path = target_dir / gazetteer_parser_name
    gazetteer_parser_metadata = []
    for ent in sorted(gazetteer_entities):
        # Fetch the compiled parser in the resources
        source_parser_path = find_gazetteer_entity_data_path(language, ent)
        short_name = get_builtin_entity_shortname(ent).lower()
        target_parser_path = gazetteer_parser_path / short_name
        parser_metadata = {
            "entity_identifier": ent,
            "entity_parser": short_name
        }
        gazetteer_parser_metadata.append(parser_metadata)
        # Copy the single entity parser
        shutil.copytree(str(source_parser_path), str(target_parser_path))
    # Dump the parser metadata
    gazetteer_entity_parser_metadata = {
        "parsers_metadata": gazetteer_parser_metadata
    }
    gazetteer_parser_metadata_path = gazetteer_parser_path / "metadata.json"
    with gazetteer_parser_metadata_path.open("w", encoding="utf-8") as f:
        f.write(json_string(gazetteer_entity_parser_metadata))
    return gazetteer_parser_name


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


def _get_caching_key(language, entity_scope):
    tuple_key = (language,)
    tuple_key += tuple(entity for entity in sorted(entity_scope))
    return tuple_key
