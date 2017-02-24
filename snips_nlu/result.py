def is_range(rng):
    if not isinstance(rng, (list, tuple)) or len(rng) != 2 or rng[0] > rng[1]:
        raise ValueError("range must be a length 2 list or tuple and must be "
                         "valid")


def intent_classification_result(intent_name=None, prob=None):
    if intent_name is None:
        return None
    res = {"intent": intent_name}
    if prob is not None:
        res["prob"] = prob
    return res


def parsed_entity(match_range, value, entity, slot_name=None, **kwargs):
    parsed_ent = dict(kwargs)

    is_range(match_range)
    parsed_ent["range"] = match_range

    assert isinstance(value, (str, unicode))
    parsed_ent["value"] = value

    assert isinstance(value, (str, unicode))
    parsed_ent["entity"] = entity

    if slot_name is not None:
        assert isinstance(value, (str, unicode))
        parsed_ent["slotName"] = slot_name

    return parsed_ent


def result(text, parsed_intent=None, parsed_entities=None):
    assert isinstance(text, (str, unicode))
    res = {"text": text, "intent": None, "entities": []}

    if parsed_intent is not None:
        if parsed_intent["intent"] is None:
            raise LookupError("'intent' key can't be None if a result is "
                              "passed")
        intent = {"name": parsed_intent["intent"]}
        if parsed_intent["prob"] is not None:
            intent["prob"] = parsed_intent["prob"]
        res["intent"] = intent

    mandatory_keys = ["range", "value", "entity"]
    if parsed_entities is not None:
        for ent in parsed_entities:
            for k in mandatory_keys:
                if k not in ent:
                    raise LookupError("Missing '%s' key" % k)
    res["entities"] = parsed_entities
    return res
