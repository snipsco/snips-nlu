from __future__ import unicode_literals

from snips_nlu.constants import (
    RES_INTENT_NAME, RES_PROBABILITY, RES_INTENT, RES_SLOTS, RES_MATCH_RANGE,
    RES_RAW_VALUE, RES_INPUT, RES_SLOT_NAME, RES_ENTITY, RES_VALUE)


def intent_classification_result(intent_name, probability):
    return {
        RES_INTENT_NAME: intent_name,
        RES_PROBABILITY: probability
    }


def _slot(match_range, value, entity, slot_name):
    """Internal slot yet to be resolved"""
    return {
        RES_MATCH_RANGE: list(match_range),
        RES_VALUE: value,
        RES_ENTITY: entity,
        RES_SLOT_NAME: slot_name
    }


def custom_slot(internal_slot):
    return {
        RES_MATCH_RANGE: internal_slot[RES_MATCH_RANGE],
        RES_RAW_VALUE: internal_slot[RES_VALUE],
        RES_VALUE: {
            "kind": "Custom",
            "value": internal_slot[RES_VALUE]
        },
        RES_ENTITY: internal_slot[RES_ENTITY],
        RES_SLOT_NAME: internal_slot[RES_SLOT_NAME]
    }


def builtin_slot(internal_slot, resolved_value):
    return {
        RES_MATCH_RANGE: list(internal_slot[RES_MATCH_RANGE]),
        RES_RAW_VALUE: internal_slot[RES_VALUE],
        RES_VALUE: resolved_value,
        RES_ENTITY: internal_slot[RES_ENTITY],
        RES_SLOT_NAME: internal_slot[RES_SLOT_NAME]
    }


def resolved_slot(match_range, raw_value, resolved_value, entity, slot_name):
    return {
        RES_MATCH_RANGE: list(match_range),
        RES_RAW_VALUE: raw_value,
        RES_VALUE: resolved_value,
        RES_ENTITY: entity,
        RES_SLOT_NAME: slot_name
    }


def parsing_result(input, intent, slots):  # pylint:disable=redefined-builtin
    return {
        RES_INPUT: input,
        RES_INTENT: intent,
        RES_SLOTS: slots
    }


def is_empty(result):
    return result[RES_INTENT] is None and result[RES_SLOTS] is None


def empty_result(input):  # pylint:disable=redefined-builtin
    return parsing_result(input=input, intent=None, slots=None)
