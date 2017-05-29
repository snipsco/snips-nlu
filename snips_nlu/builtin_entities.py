from __future__ import unicode_literals

from collections import defaultdict

from enum import Enum
from rustling import RustlingError
from rustling import RustlingParser as _RustlingParser

from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY, LABEL, \
    RUSTLING_DIM_KIND, SUPPORTED_LANGUAGES
from snips_nlu.languages import Language
from utils import LimitedSizeDict, classproperty


class BuiltInEntity(Enum):
    NUMBER = {
        LABEL: "snips/number",
        RUSTLING_DIM_KIND: "number",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    ORDINAL = {
        LABEL: "snips/ordinal",
        RUSTLING_DIM_KIND: "ordinal",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    TEMPERATURE = {
        LABEL: "snips/temperature",
        RUSTLING_DIM_KIND: "temperature",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    DATETIME = {
        LABEL: "snips/datetime",
        RUSTLING_DIM_KIND: "time",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
        }
    }

    DURATION = {
        LABEL: "snips/duration",
        RUSTLING_DIM_KIND: "duration",
        SUPPORTED_LANGUAGES: {
            Language.EN,
            Language.FR,
            Language.ES
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


def scope_to_dim_kinds(scope):
    return [entity.rustling_dim_kind for entity in scope]


_SUPPORTED_DIM_KINDS_BY_LANGUAGE = defaultdict(set)
for entity in BuiltInEntity:
    for language in entity.supported_languages:
        _SUPPORTED_DIM_KINDS_BY_LANGUAGE[language].add(
            entity.rustling_dim_kind)


class RustlingParser(object):
    def __init__(self, language):
        self.language = language
        self.parser = _RustlingParser(language.rustling_code)
        self._cache = LimitedSizeDict(size_limit=1000)
        self.supported_entities = set(
            _SUPPORTED_DIM_KINDS_BY_LANGUAGE[self.language])

    def parse(self, text):
        text = text.lower()  # Rustling only work with lowercase
        if text not in self._cache:
            try:
                parser_result = self.parser.parse(text)
            except RustlingError:
                parser_result = []
            self._cache[text] = parser_result
        return self._cache[text]

    def supports_dim_kind(self, dim_kind):
        return dim_kind in self.supported_entities


_RUSTLING_PARSERS = dict()
for language in Language:
    try:
        _RUSTLING_PARSERS[language] = RustlingParser(language)
    except RustlingError:
        pass

RUSTLING_SUPPORTED_LANGUAGES = set(_RUSTLING_PARSERS.keys())

RUSTLING_DIM_KINDS = {
    "number",
    "ordinal",
    "temperature",
    "time",
    "duration"
}


def get_builtin_entities(text, language, scope=None):
    global _RUSTLING_CACHE
    global _RUSTLING_PARSERS

    parser = _RUSTLING_PARSERS.get(language, False)
    if not parser:
        return []

    if scope is None:
        dim_kinds = set(RUSTLING_DIM_KINDS)
    else:
        dim_kinds = scope_to_dim_kinds(scope)

    # Don't detect entities that are not supportBuiltInEntity
    dim_kinds = [d for d in dim_kinds if parser.supports_dim_kind(d)]

    parsed_entities = []
    for ent in parser.parse(text):
        if ent["dim"] in dim_kinds:
            parsed_entity = {
                MATCH_RANGE: (ent["char_range"]["start"],
                              ent["char_range"]["end"]),
                VALUE: ent["value"],
                ENTITY: BuiltInEntity.from_rustling_dim_kind(ent["dim"])
            }
            parsed_entities.append(parsed_entity)
    return parsed_entities


def is_builtin_entity(entity_label):
    return entity_label in BuiltInEntity.built_in_entity_by_label
