from utils import namedtuple_with_defaults

IntentClassificationResult = namedtuple_with_defaults(
    "IntentClassificationResult", "name prob")


def is_range(rng):
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise ValueError("range must be a length 2 list or tuple")


def parsed_entity(match_range, value, entity, role=None, **kwargs):
    parsed_ent = dict()

    is_range(match_range)
    parsed_ent["range"] = match_range

    assert isinstance(value, [str, unicode])
    parsed_ent["value"] = value

    assert isinstance(value, [str, unicode])
    parsed_ent["entity"] = entity

    if role is not None:
        assert isinstance(value, [str, unicode])
        parsed_ent["role"] = role

    parsed_ent.update(kwargs)
    return parsed_ent


def result(text, intent=None, entities=None):
    assert isinstance(text, (str, unicode))
    res = {"text": text, "intent": {}, "entities": []}

    if intent is not None:
        assert isinstance(intent, IntentClassificationResult)
        if intent.name is not None:
            res["intent"]["name"] = intent.name
            if intent.prob is not None:
                res["intent"]["prob"] = intent.prob

    mandatory_keys = ["range", "value", "entity"]
    if entities is not None:
        for ent in entities:
            for k in mandatory_keys:
                if k not in ent:
                    raise LookupError("Missing %s key" % k)
    res["entities"] = entities
    return res
