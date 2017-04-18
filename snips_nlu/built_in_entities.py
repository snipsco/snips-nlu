from duckling import core
from enum import Enum

from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY, LABEL, DUCKLING_DIM
from utils import LimitedSizeDict, classproperty


class BuiltInEntity(Enum):
    DATETIME = {LABEL: "snips/datetime", DUCKLING_DIM: "time"}

    DURATION = {LABEL: "snips/duration", DUCKLING_DIM: "duration"}

    TIME_CYCLE = {LABEL: "snips/time-cycle", DUCKLING_DIM: "cycle"}

    NUMBER = {LABEL: "snips/number", DUCKLING_DIM: "number"}

    ORDINAL = {LABEL: "snips/ordinal", DUCKLING_DIM: "ordinal"}

    TEMPERATURE = {LABEL: "snips/temperature", DUCKLING_DIM: "temperature"}

    UNIT = {LABEL: "snips/unit", DUCKLING_DIM: "unit"}

    AMOUNT_OF_MONEY = {
        LABEL: "snips/amount-of-money",
        DUCKLING_DIM: "amount-of-money"
    }

    @property
    def label(self):
        return self.value[LABEL]

    @property
    def duckling_dim(self):
        return self.value[DUCKLING_DIM]

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
    def built_in_entity_by_duckling_dim(cls):
        try:
            return cls._built_in_entity_by_duckling_dim
        except AttributeError:
            cls._built_in_entity_by_duckling_dim = dict()
            for ent in cls:
                cls._built_in_entity_by_duckling_dim[ent.duckling_dim] = ent
        return cls._built_in_entity_by_duckling_dim

    @classmethod
    def from_duckling_dim(cls, duckling_dim, default=None):
        try:
            ent = cls.built_in_entity_by_duckling_dim[duckling_dim]
        except KeyError:
            if default is None:
                raise KeyError("Unknown duckling dim '%s'" % duckling_dim)
            else:
                return default
        return ent


def scope_to_dims(scope):
    return [entity.duckling_dim for entity in scope]


_DUCKLING_CACHE = LimitedSizeDict(size_limit=1000)


def get_built_in_entities(text, language, scope=None):
    global _DUCKLING_CACHE
    language = language.duckling_code
    if scope is None:
        dims = core.get_dims(language)
    else:
        dims = scope_to_dims(scope)

    # Don't detect entities that are not BuiltInEntity
    dims = [d for d in dims if
            d in BuiltInEntity.built_in_entity_by_duckling_dim]

    if (text, language) not in _DUCKLING_CACHE:
        parse = core.parse(language, text)
        _DUCKLING_CACHE[(text, language)] = parse
    else:
        parse = _DUCKLING_CACHE[(text, language)]

    parsed_entities = []
    for ent in parse:
        if ent["dim"] in dims:
            parsed_entity = {
                MATCH_RANGE: (ent["start"], ent["end"]),
                VALUE: ent["body"],
                ENTITY: BuiltInEntity.from_duckling_dim(ent["dim"])
            }
            parsed_entities.append(parsed_entity)
    return parsed_entities


def clear_cache():
    _DUCKLING_CACHE.clear()
