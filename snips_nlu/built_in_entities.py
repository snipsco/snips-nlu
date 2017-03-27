from duckling import core
from enum import Enum

core.load()


class BuiltInEntity(Enum):
    DATETIME = {"label": "snips/datetime", "duckling_dim": "time"}
    DURATION = {"label": "snips/duration", "duckling_dim": "duration"}
    NUMBER = {"label": "snips/number", "duckling_dim": "number"}


DUCKLING_DIM_TO_SNIPS_ENTITY = dict((ent.value["duckling_dim"], ent)
                                    for ent in BuiltInEntity)


def scope_to_dims(scope):
    return [entity.value["duckling_dim"] for entity in scope]


def get_built_in_entities(text, language, scope=None):
    if scope is None:
        scope = []
    dims = scope_to_dims(scope)
    parse = core.parse(language, text, dims=dims)
    return [
        {
            "match_range": (ent["start"], ent["end"]),
            "value": ent["body"],
            "entity": DUCKLING_DIM_TO_SNIPS_ENTITY[ent["dim"]]
        } for ent in parse]


if __name__ == '__main__':
    texts = ["be there at 2p.m. please", "we go there at 2 is it OK"]
    for t in texts:
        entities = get_built_in_entities(t, "en", scope=[])
        print t, entities
