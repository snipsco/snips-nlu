from duckling import core
from enum import Enum

from utils import LimitedSizeDict

core.load()


class BuiltInEntity(Enum):
    DATETIME = {"label": "snips/datetime", "duckling_dim": "time"}
    DURATION = {"label": "snips/duration", "duckling_dim": "duration"}
    NUMBER = {"label": "snips/number", "duckling_dim": "number"}


DUCKLING_DIM_TO_SNIPS_ENTITY = dict((ent.value["duckling_dim"], ent)
                                    for ent in BuiltInEntity)
BUILT_ENTITY_BY_LABEL = dict((ent.value["label"], ent)
                             for ent in BuiltInEntity)


def get_built_in_by_label(label):
    return BUILT_ENTITY_BY_LABEL[label]


def scope_to_dims(scope):
    return [entity.value["duckling_dim"] for entity in scope]


_DUCKLING_CACHE = LimitedSizeDict(size_limit=1000)


def get_built_in_entities(text, language, scope=None):
    global _DUCKLING_CACHE
    if scope is None:
        dims = core.get_dims(language)
    else:
        dims = scope_to_dims(scope)
        
    if text not in _DUCKLING_CACHE:
        parse = core.parse(language, text)
        _DUCKLING_CACHE[(text, language)] = parse
    else:
        parse = _DUCKLING_CACHE[(text, language)]

    parsed_entities = []
    for ent in parse:
        if ent["dim"] in dims:
            parsed_entity = {
                "match_range": (ent["start"], ent["end"]),
                "value": ent["body"],
                "entity": DUCKLING_DIM_TO_SNIPS_ENTITY[ent["dim"]]
            }
            parsed_entities.append(parsed_entity)
    return parsed_entities


if __name__ == '__main__':
    texts = ["be there at 2p.m. please", "we go there at 2 is it OK"]
    for t in texts:
        entities = get_built_in_entities(t, "en", scope=[])
        print t, entities
