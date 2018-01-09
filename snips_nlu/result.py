from snips_nlu.constants import (
    RES_INTENT_NAME, RES_PROBABILITY, RES_INTENT, RES_SLOTS, RES_MATCH_RANGE,
    RES_RAW_VALUE, RES_INPUT, RES_SLOT_NAME, RES_ENTITY, RES_VALUE)


def intent_classification_result(intent_name, probability):
    return {
        RES_INTENT_NAME: intent_name,
        RES_PROBABILITY: probability
    }


def internal_slot(match_range, value, entity, slot_name):
    return {
        RES_MATCH_RANGE: list(match_range),
        RES_VALUE: value,
        RES_ENTITY: entity,
        RES_SLOT_NAME: slot_name
    }


def custom_slot(internal_slot_):
    return {
        RES_MATCH_RANGE: internal_slot_[RES_MATCH_RANGE],
        RES_RAW_VALUE: internal_slot_[RES_VALUE],
        RES_VALUE: {
            "kind": "Custom",
            "value": internal_slot_[RES_VALUE]
        },
        RES_ENTITY: internal_slot_[RES_ENTITY],
        RES_SLOT_NAME: internal_slot_[RES_SLOT_NAME]
    }


def parsing_result(input, intent, slots):  # pylint:disable=redefined-builtin
    return {
        RES_INPUT: input,
        RES_INTENT: intent,
        RES_SLOTS: slots
    }


def is_empty(result):
    return result[RES_INTENT] is None and result[RES_SLOTS] is None


def empty_result(input_):
    return parsing_result(input=input_, intent=None, slots=None)
