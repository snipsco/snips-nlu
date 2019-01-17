from __future__ import unicode_literals

from snips_nlu.constants import (
    RES_ENTITY, RES_INPUT, RES_INTENT, RES_INTENT_NAME, RES_MATCH_RANGE,
    RES_PROBA, RES_RAW_VALUE, RES_SLOTS, RES_SLOT_NAME, RES_VALUE, ENTITY_KIND,
    RESOLVED_VALUE, VALUE)


def intent_classification_result(intent_name, probability):
    """Creates an intent classification result to be returned by
    :meth:`.IntentClassifier.get_intent`

    Example:

        >>> intent_classification_result("GetWeather", 0.93)
        {'intentName': 'GetWeather', 'probability': 0.93}
    """
    return {
        RES_INTENT_NAME: intent_name,
        RES_PROBA: probability
    }


def unresolved_slot(match_range, value, entity, slot_name):
    """Creates an internal slot yet to be resolved

    Example:

        >>> from snips_nlu.common.utils import json_string
        >>> slot = unresolved_slot([0, 8], "tomorrow", "snips/datetime", \
            "startDate")
        >>> print(json_string(slot, indent=4, sort_keys=True))
        {
            "entity": "snips/datetime",
            "range": {
                "end": 8,
                "start": 0
            },
            "slotName": "startDate",
            "value": "tomorrow"
        }
    """
    return {
        RES_MATCH_RANGE: _convert_range(match_range),
        RES_VALUE: value,
        RES_ENTITY: entity,
        RES_SLOT_NAME: slot_name
    }


def custom_slot(internal_slot, resolved_value=None):
    """Creates a custom slot with *resolved_value* being the reference value
    of the slot

    Example:

        >>> s = unresolved_slot([10, 19], "earl grey", "beverage", "beverage")
        >>> from snips_nlu.common.utils import json_string
        >>> print(json_string(custom_slot(s, "tea"), indent=4, sort_keys=True))
        {
            "entity": "beverage",
            "range": {
                "end": 19,
                "start": 10
            },
            "rawValue": "earl grey",
            "slotName": "beverage",
            "value": {
                "kind": "Custom",
                "value": "tea"
            }
        }
    """

    if resolved_value is None:
        resolved_value = internal_slot[RES_VALUE]
    return {
        RES_MATCH_RANGE: _convert_range(internal_slot[RES_MATCH_RANGE]),
        RES_RAW_VALUE: internal_slot[RES_VALUE],
        RES_VALUE: {
            "kind": "Custom",
            "value": resolved_value
        },
        RES_ENTITY: internal_slot[RES_ENTITY],
        RES_SLOT_NAME: internal_slot[RES_SLOT_NAME]
    }


def builtin_slot(internal_slot, resolved_value):
    """Creates a builtin slot with *resolved_value* being the resolved value
    of the slot

    Example:

        >>> rng = [10, 32]
        >>> raw_value = "twenty degrees celsius"
        >>> entity = "snips/temperature"
        >>> slot_name = "beverageTemperature"
        >>> s = unresolved_slot(rng, raw_value, entity, slot_name)
        >>> resolved = {
        ...     "kind": "Temperature",
        ...     "value": 20,
        ...     "unit": "celsius"
        ... }
        >>> from snips_nlu.common.utils import json_string
        >>> print(json_string(builtin_slot(s, resolved), indent=4))
        {
            "entity": "snips/temperature",
            "range": {
                "end": 32,
                "start": 10
            },
            "rawValue": "twenty degrees celsius",
            "slotName": "beverageTemperature",
            "value": {
                "kind": "Temperature",
                "unit": "celsius",
                "value": 20
            }
        }
    """
    return {
        RES_MATCH_RANGE: _convert_range(internal_slot[RES_MATCH_RANGE]),
        RES_RAW_VALUE: internal_slot[RES_VALUE],
        RES_VALUE: resolved_value,
        RES_ENTITY: internal_slot[RES_ENTITY],
        RES_SLOT_NAME: internal_slot[RES_SLOT_NAME]
    }


def resolved_slot(match_range, raw_value, resolved_value, entity, slot_name):
    """Creates a resolved slot

    Args:
        match_range (dict): Range of the slot within the sentence
            (ex: {"start": 3, "end": 10})
        raw_value (str): Slot value as it appears in the sentence
        resolved_value (dict): Resolved value of the slot
        entity (str): Entity which the slot belongs to
        slot_name (str): Slot type

    Returns:
        dict: The resolved slot

    Example:

        >>> resolved_value = {
        ...     "kind": "Temperature",
        ...     "value": 20,
        ...     "unit": "celsius"
        ... }
        >>> slot = resolved_slot({"start": 10, "end": 19}, "earl grey",
        ... resolved_value, "beverage", "beverage")
        >>> from snips_nlu.common.utils import json_string
        >>> print(json_string(slot, indent=4, sort_keys=True))
        {
            "entity": "beverage",
            "range": {
                "end": 19,
                "start": 10
            },
            "rawValue": "earl grey",
            "slotName": "beverage",
            "value": {
                "kind": "Temperature",
                "unit": "celsius",
                "value": 20
            }
        }
    """
    return {
        RES_MATCH_RANGE: match_range,
        RES_RAW_VALUE: raw_value,
        RES_VALUE: resolved_value,
        RES_ENTITY: entity,
        RES_SLOT_NAME: slot_name
    }


def parsing_result(input, intent, slots):  # pylint:disable=redefined-builtin
    """Create the final output of :meth:`.SnipsNLUEngine.parse` or
    :meth:`.IntentParser.parse`

    Example:

        >>> text = "Hello Bill!"
        >>> intent_result = intent_classification_result("Greeting", 0.95)
        >>> internal_slot = unresolved_slot([6, 10], "Bill", "name",
        ... "greetee")
        >>> slots = [custom_slot(internal_slot, "William")]
        >>> res = parsing_result(text, intent_result, slots)
        >>> from snips_nlu.common.utils import json_string
        >>> print(json_string(res, indent=4, sort_keys=True))
        {
            "input": "Hello Bill!",
            "intent": {
                "intentName": "Greeting",
                "probability": 0.95
            },
            "slots": [
                {
                    "entity": "name",
                    "range": {
                        "end": 10,
                        "start": 6
                    },
                    "rawValue": "Bill",
                    "slotName": "greetee",
                    "value": {
                        "kind": "Custom",
                        "value": "William"
                    }
                }
            ]
        }
    """
    return {
        RES_INPUT: input,
        RES_INTENT: intent,
        RES_SLOTS: slots
    }


def extraction_result(intent, slots):
    """Create the items in the output of :meth:`.SnipsNLUEngine.parse` or
    :meth:`.IntentParser.parse` when called with a defined ``top_n`` value

    This differs from :func:`.parsing_result` in that the input is omitted.

    Example:

        >>> intent_result = intent_classification_result("Greeting", 0.95)
        >>> internal_slot = unresolved_slot([6, 10], "Bill", "name",
        ... "greetee")
        >>> slots = [custom_slot(internal_slot, "William")]
        >>> res = extraction_result(intent_result, slots)
        >>> from snips_nlu.common.utils import json_string
        >>> print(json_string(res, indent=4, sort_keys=True))
        {
            "intent": {
                "intentName": "Greeting",
                "probability": 0.95
            },
            "slots": [
                {
                    "entity": "name",
                    "range": {
                        "end": 10,
                        "start": 6
                    },
                    "rawValue": "Bill",
                    "slotName": "greetee",
                    "value": {
                        "kind": "Custom",
                        "value": "William"
                    }
                }
            ]
        }
    """
    return {
        RES_INTENT: intent,
        RES_SLOTS: slots
    }


def is_empty(result):
    """Check if a result is empty

    Example:

        >>> res = empty_result("foo bar", 1.0)
        >>> is_empty(res)
        True
    """
    return result[RES_INTENT][RES_INTENT_NAME] is None


def empty_result(input, probability):  # pylint:disable=redefined-builtin
    """Creates an empty parsing result of the same format as the one of
    :func:`parsing_result`

    An empty is typically returned by a :class:`.SnipsNLUEngine` or
    :class:`.IntentParser` when no intent nor slots were found.

    Example:

        >>> res = empty_result("foo bar", 0.8)
        >>> from snips_nlu.common.utils import json_string
        >>> print(json_string(res, indent=4, sort_keys=True))
        {
            "input": "foo bar",
            "intent": {
                "intentName": null,
                "probability": 0.8
            },
            "slots": []
        }
    """
    intent = intent_classification_result(None, probability)
    return parsing_result(input=input, intent=intent, slots=[])


def parsed_entity(entity_kind, entity_value, entity_resolved_value,
                  entity_range):
    """Create the items in the output of
        :meth:`snips_nlu.entity_parser.EntityParser.parse`

    Example:
        >>> resolved_value = dict(age=28, role="datascientist")
        >>> range = dict(start=0, end=6)
        >>> ent = parsed_entity("snipster", "adrien", resolved_value, range)
        >>> import json
        >>> print(json.dumps(ent, indent=4, sort_keys=True))
        {
            "entity_kind": "snipster",
            "range": {
                "end": 6,
                "start": 0
            },
            "resolved_value": {
                "age": 28,
                "role": "datascientist"
            },
            "value": "adrien"
        }
    """
    return {
        VALUE: entity_value,
        RESOLVED_VALUE: entity_resolved_value,
        ENTITY_KIND: entity_kind,
        RES_MATCH_RANGE: entity_range
    }


def _convert_range(rng):
    if isinstance(rng, dict):
        return rng
    return {
        "start": rng[0],
        "end": rng[1]
    }
