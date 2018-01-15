from __future__ import unicode_literals
from builtins import str
from builtins import object

from collections import defaultdict

from enum import Enum
from rustling import (RustlingParser as _RustlingParser, RustlingError,
                      all_configs)

from snips_nlu.constants import RES_MATCH_RANGE, VALUE, ENTITY, LABEL, \
    RUSTLING_DIM_KIND, SUPPORTED_LANGUAGES
from snips_nlu.languages import Language
from snips_nlu.utils import LimitedSizeDict, classproperty


class BuiltInEntity(Enum):
    NUMBER = {
        LABEL: "snips/number",
        RUSTLING_DIM_KIND: "Number",
        SUPPORTED_LANGUAGES: {
            Language.DE,
            Language.EN,
            Language.ES,
            Language.FR,
            Language.KO,
        }
    }

    ORDINAL = {
        LABEL: "snips/ordinal",
        RUSTLING_DIM_KIND: "Ordinal",
        SUPPORTED_LANGUAGES: {
            Language.DE,
            Language.EN,
            Language.ES,
            Language.FR,
            Language.KO,
        }
    }

    PERCENTAGE = {
        LABEL: "snips/percentage",
        RUSTLING_DIM_KIND: "Percentage",
        SUPPORTED_LANGUAGES: {
            Language.DE,
            Language.EN,
            Language.ES,
            Language.FR,
        }
    }

    TEMPERATURE = {
        LABEL: "snips/temperature",
        RUSTLING_DIM_KIND: "Temperature",
        SUPPORTED_LANGUAGES: {
            Language.DE,
            Language.EN,
            Language.ES,
            Language.FR,
            Language.KO,
        }
    }

    DATETIME = {
        LABEL: "snips/datetime",
        RUSTLING_DIM_KIND: "Time",
        SUPPORTED_LANGUAGES: {
            Language.DE,
            Language.EN,
            Language.ES,
            Language.FR,
            Language.KO,
        }
    }

    DURATION = {
        LABEL: "snips/duration",
        RUSTLING_DIM_KIND: "Duration",
        SUPPORTED_LANGUAGES: {
            Language.DE,
            Language.EN,
            Language.ES,
            Language.FR,
            Language.KO,
        }
    }

    AMOUNT_OF_MONEY = {
        LABEL: "snips/amountOfMoney",
        RUSTLING_DIM_KIND: "AmountOfMoney",
        SUPPORTED_LANGUAGES: {
            Language.DE,
            Language.EN,
            Language.ES,
            Language.FR,
            Language.KO,
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


_RUSTLING_SUPPORTED_BUILTINS_BY_LANGUAGE = dict()

for k, v in all_configs().items():
    try:
        lang = Language.from_rustling_code(k)
    except KeyError:
        continue
    _RUSTLING_SUPPORTED_BUILTINS_BY_LANGUAGE[lang] = \
        set(BuiltInEntity.from_rustling_dim_kind(e) for e in v)

_SUPPORTED_BUILTINS_BY_LANGUAGE = defaultdict(set)
for builtin_entity in BuiltInEntity:
    for lang in builtin_entity.supported_languages:
        if builtin_entity not in \
                _RUSTLING_SUPPORTED_BUILTINS_BY_LANGUAGE[lang]:
            raise KeyError("Found '%s' in supported languages of '%s' but, "
                           "'%s' is not supported in rustling.all_configs()" %
                           (lang, builtin_entity, lang))
        _SUPPORTED_BUILTINS_BY_LANGUAGE[lang].add(builtin_entity)

RUSTLING_ENTITIES = set(
    kind for kinds in _RUSTLING_SUPPORTED_BUILTINS_BY_LANGUAGE.values()
    for kind in kinds)

_DIM_KIND_TO_ENTITY = {e.rustling_dim_kind: e for e in RUSTLING_ENTITIES}


def scope_to_dim_kinds(scope):
    return [entity.rustling_dim_kind for entity in scope]


class RustlingParser(object):
    def __init__(self, language):
        self.language = language
        self.parser = _RustlingParser(language.rustling_code)
        self._cache = LimitedSizeDict(size_limit=1000)
        self.supported_entities = set(
            _SUPPORTED_BUILTINS_BY_LANGUAGE[self.language])

    def parse(self, text, scope=None):
        text = text.lower()  # Rustling only work with lowercase
        if scope is not None:
            scope = [e.rustling_dim_kind for e in scope]
        cache_key = (text, str(scope))
        if cache_key not in self._cache:
            try:
                if scope is None:
                    parser_result = self.parser.parse(text)
                else:
                    parser_result = self.parser.parse_with_kind_order(
                        text, scope)
            except RustlingError:
                parser_result = []
            self._cache[cache_key] = parser_result
        return self._cache[cache_key]

    def supports_entity(self, entity):
        return entity in self.supported_entities


_RUSTLING_PARSERS = dict()
for lang in Language:
    try:
        _RUSTLING_PARSERS[lang] = RustlingParser(lang)
    except RustlingError:
        pass


def get_supported_builtin_entities(language):
    return _SUPPORTED_BUILTINS_BY_LANGUAGE[language]


def get_builtin_entities(text, language, scope=None):
    global _RUSTLING_PARSERS

    parser = _RUSTLING_PARSERS.get(language, False)
    if not parser:
        return []

    if scope is None:
        scope = set(RUSTLING_ENTITIES)

    # Don't detect entities that are not supported BuiltInEntity
    # a entity can be supported in Rustling but we may want not to support it
    entities = [e for e in scope if parser.supports_entity(e)]
    entities_parsed_dims = set(e.rustling_dim_kind for e in entities)
    parsed_entities = []
    for entity in parser.parse(text, scope=scope):
        if entity["dim"] in entities_parsed_dims:
            parsed_entity = {
                RES_MATCH_RANGE: (entity["char_range"]["start"],
                                  entity["char_range"]["end"]),
                VALUE: entity["value"],
                ENTITY: _DIM_KIND_TO_ENTITY[entity["dim"]]
            }
            parsed_entities.append(parsed_entity)
    return parsed_entities


def is_builtin_entity(entity_label):
    # pylint: disable=E1135
    return entity_label in BuiltInEntity.built_in_entity_by_label
