from __future__ import unicode_literals

from collections import defaultdict

import rustling
from enum import Enum
from rustling import RustlingParser as _RustlingParser, RustlingError

from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY, LABEL, \
    RUSTLING_DIM_KIND, SUPPORTED_LANGUAGES
from snips_nlu.languages import Language
from utils import LimitedSizeDict, classproperty


class BuiltInEntity(Enum):
    NUMBER = {
        LABEL: "snips/number",
        RUSTLING_DIM_KIND: "Number",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    TEMPERATURE = {
        LABEL: "snips/temperature",
        RUSTLING_DIM_KIND: "Temperature",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    DATETIME = {
        LABEL: "snips/datetime",
        RUSTLING_DIM_KIND: "Time",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    DURATION = {
        LABEL: "snips/duration",
        RUSTLING_DIM_KIND: "Duration",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    AMOUNT_OF_MONEY = {
        LABEL: "snips/amountOfMoney",
        RUSTLING_DIM_KIND: "AmountOfMoney",
        SUPPORTED_LANGUAGES: {
            Language.EN
        }
    }

    @property
    def label(self):
        return self.value[LABEL]

    @property
    def rustling_dim_kind(self):
        return self.value[RUSTLING_DIM_KIND]

    @property
    def supported_languages(self):
        return self.value[SUPPORTED_LANGUAGES]

    @classproperty
    @classmethod
    def built_in_entity_by_label(cls):
        try:
            return cls._built_in_entity_by_label
        except AttributeError:
            cls._built_in_entity_by_label = dict()
            for ent in cls:
                cls._built_in_entity_by_label[ent.label] = ent
        return cls._built_in_entity_by_label

    @classmethod
    def from_label(cls, label, default=None):
        try:
            ent = cls.built_in_entity_by_label[label]
        except KeyError:
            if default is None:
                raise KeyError("Unknown entity '%s'" % label)
            else:
                return default
        return ent

    @classproperty
    @classmethod
    def built_in_entity_by_rustling_dim_kind(cls):
        try:
            return cls._built_in_entity_by_rustling_dim_kind
        except AttributeError:
            cls._built_in_entity_by_rustling_dim_kind = dict()
            for ent in cls:
                cls._built_in_entity_by_rustling_dim_kind[
                    ent.rustling_dim_kind] = ent
        return cls._built_in_entity_by_rustling_dim_kind

    @classmethod
    def from_rustling_dim_kind(cls, rustling_dim_kind, default=None):
        try:
            ent = cls.built_in_entity_by_rustling_dim_kind[rustling_dim_kind]
        except KeyError:
            if default is None:
                raise KeyError(
                    "Unknown rustling dim kind '%s'" % rustling_dim_kind)
            else:
                return default
        return ent


_RUSTLING_SUPPORTED_BUILTINS_BY_LANGUAGE = {
    Language.from_rustling_code(k.upper()): set(
        BuiltInEntity.from_rustling_dim_kind(e) for e in v)
    for k, v in rustling.all_configs().iteritems()
}

_SUPPORTED_BUILTINS_BY_LANGUAGE = defaultdict(set)
for entity in BuiltInEntity:
    for language in entity.supported_languages:
        if not entity in _RUSTLING_SUPPORTED_BUILTINS_BY_LANGUAGE[language]:
            raise KeyError("Found '%s' in supported languages of '%s' but, "
                           "'%s' is not supported in rustling.all_configs()" %
                           (language, entity, language))
        _SUPPORTED_BUILTINS_BY_LANGUAGE[language].add(entity)

RUSTLING_ENTITIES = set(
    kind for kinds in _RUSTLING_SUPPORTED_BUILTINS_BY_LANGUAGE.values()
    for kind in kinds)

_ENTITY_TO_PARSED_DIM_KIND = dict()
for entity in RUSTLING_ENTITIES:
    parsed_dim_kind_name = ""
    for i, char in enumerate(entity.rustling_dim_kind):
        lowered = char.lower()
        if lowered != char and i != 0:
            parsed_dim_kind_name += "-"
        parsed_dim_kind_name += lowered
    _ENTITY_TO_PARSED_DIM_KIND[entity] = parsed_dim_kind_name

_PARSED_DIM_KIND_TO_ENTITY = {
    v: k
    for k, v in _ENTITY_TO_PARSED_DIM_KIND.iteritems()
}


def scope_to_dim_kinds(scope):
    return [_ENTITY_TO_PARSED_DIM_KIND[entity] for entity in scope]


class RustlingParser(object):
    def __init__(self, language):
        self.language = language
        self.parser = _RustlingParser(language.rustling_code)
        self._cache = LimitedSizeDict(size_limit=1000)
        self.supported_entities = set(
            _SUPPORTED_BUILTINS_BY_LANGUAGE[self.language])

    def parse(self, text):
        text = text.lower()  # Rustling only work with lowercase
        if text not in self._cache:
            try:
                parser_result = self.parser.parse(text)
            except RustlingError:
                parser_result = []
            self._cache[text] = parser_result
        return self._cache[text]

    def supports_entity(self, entity):
        return entity in self.supported_entities


_RUSTLING_PARSERS = dict()
for language in Language:
    try:
        _RUSTLING_PARSERS[language] = RustlingParser(language)
    except RustlingError:
        pass

RUSTLING_SUPPORTED_LANGUAGES = set(_RUSTLING_PARSERS.keys())


def get_builtin_entities(text, language, scope=None):
    global _RUSTLING_CACHE
    global _RUSTLING_PARSERS

    parser = _RUSTLING_PARSERS.get(language, False)
    if not parser:
        return []

    if scope is None:
        entities = set(RUSTLING_ENTITIES)
    else:
        entities = scope_to_dim_kinds(scope)

    # Don't detect entities that are not supportBuiltInEntity
    entities = [e for e in entities if parser.supports_entity(e)]
    entities_parsed_dims = set(_ENTITY_TO_PARSED_DIM_KIND[e] for e in entities)
    parsed_entities = []
    for ent in parser.parse(text):
        if ent["dim"] in entities_parsed_dims:
            parsed_entity = {
                MATCH_RANGE: (ent["char_range"]["start"],
                              ent["char_range"]["end"]),
                VALUE: ent["value"],
                ENTITY: _PARSED_DIM_KIND_TO_ENTITY[ent["dim"]]
            }
            parsed_entities.append(parsed_entity)
    return parsed_entities


def is_builtin_entity(entity_label):
    return entity_label in BuiltInEntity.built_in_entity_by_label
