from __future__ import unicode_literals

from snips_nlu.constants import (
    RES_INTENT_NAME, RES_PROBABILITY, RES_INTENT, RES_SLOTS, RES_MATCH_RANGE,
    RES_RAW_VALUE, RES_INPUT, RES_SLOT_NAME, RES_ENTITY, RES_VALUE)


def intent_classification_result(intent_name, probability):
    """Creates an intent classification result to be returned by
        :meth:`.IntentClassifier.get_intent`

    Example:

        >>> intent_classification_result("GetWeather", 0.93)
        {
            "intentName": "GetWeather",
            "probability": 0.93
        }
    """
    return {
        RES_INTENT_NAME: intent_name,
        RES_PROBABILITY: probability
    }


def unresolved_slot(match_range, value, entity, slot_name):
    """Creates an internal slot yet to be resolved

    Example:

        >>> unresolved_slot([0, 8], "tomorrow", "snips/datetime", "startDate")
        {
            "value": "tomorrow",
            "range": {
                "start": 0,
                "end": 8
            },
            "entity": "snips/datetime",
            "slotName": "startDate"
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
        >>> custom_slot(s, "tea")
        {
            "rawValue": "earl grey",
            "value": {
                "kind": "Custom",
                "value": "tea"
            },
            "range": {
                "start": 10,
                "end": 19
            },
            "entity": "beverage",
            "slotName": "beverage"
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
        >>> builtin_slot(s, resolved)
        {
            "rawValue": "earl grey",
            "value": {
                "kind": "Temperature",
                "value": 20,
                "unit": "celsius"
            },
            "range": {
                "start": 10,
                "end": 19
            },
            "entity": "beverage",
            "slotName": "beverage"
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
        match_range (range or tuple): Range of the slot within the sentence
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
        >>> resolved_slot((10, 19), "earl grey", resolved_value, "beverage",
        ... "beverage")
        {
            "rawValue": "earl grey",
            "value": {
                "kind": "Temperature",
                "value": 20,
                "unit": "celsius"
            },
            "range": {
                "start": 10,
                "end": 19
            },
            "entity": "beverage",
            "slotName": "beverage"
        }
    """
    return {
        RES_MATCH_RANGE: _convert_range(match_range),
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
          >>> internal_slot = unresolved_slot([6, 10], "John", "name",
          ... "greetee")
          >>> slots = [custom_slot(internal_slot, "William")]
          >>> parsing_result(text, intent_result, slots)
          {
            "input": "Hello Bill!",
            "intent": {
                "intentName": "Greeting",
                "probability": 0.95
            },
            "slots: [{
                "rawValue": "Bill",
                "value": {
                    "kind": "Custom",
                    "value": "William",
                },
                "range": {
                    "start": 6,
                    "end": 10
                },
                "entity": "name",
                "slotName": "greetee"
            }]
          }
    """
    return {
        RES_INPUT: input,
        RES_INTENT: intent,
        RES_SLOTS: slots
    }


def is_empty(result):
    """Check if a result is empty

    Example:

        >>> res = empty_result("foo bar")
        >>> is_empty(res)
        True
    """
    return result[RES_INTENT] is None and result[RES_SLOTS] is None


def empty_result(input):  # pylint:disable=redefined-builtin
    """Creates an empty parsing result of the same format as the one of
        :func:`parsing_result`

    An empty is typically returned by a :class:`.SnipsNLUEngine` or
    :class:`.IntentParser` when no intent nor slots were found.

    Example:

        >>> empty_result("foo bar")
        {
            "input": "foo bar",
            "intent": None,
            "slots": None
        }
    """
    return parsing_result(input=input, intent=None, slots=None)


def _convert_range(rng):
    if isinstance(rng, dict):
        return rng
    return {
        "start": rng[0],
        "end": rng[1]
    }
