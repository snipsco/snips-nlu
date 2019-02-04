from snips_nlu.constants import INTENTS, UTTERANCES, DATA, SLOT_NAME, ENTITY
from snips_nlu.exceptions import DatasetFormatError


def type_error(expected_type, found_type, object_label=None):
    if object_label is None:
        raise DatasetFormatError("Invalid type: expected %s but found %s"
                                 % (expected_type, found_type))
    raise DatasetFormatError("Invalid type for '%s': expected %s but found %s"
                             % (object_label, expected_type, found_type))


def validate_type(obj, expected_type, object_label=None):
    if not isinstance(obj, expected_type):
        type_error(expected_type, type(obj), object_label)


def missing_key_error(key, object_label=None):
    if object_label is None:
        raise DatasetFormatError("Missing key: '%s'" % key)
    raise DatasetFormatError("Expected %s to have key: '%s'"
                             % (object_label, key))


def validate_key(obj, key, object_label=None):
    if key not in obj:
        missing_key_error(key, object_label)


def validate_keys(obj, keys, object_label=None):
    for key in keys:
        validate_key(obj, key, object_label)


def get_slot_name_mapping(dataset, intent):
    """Returns a dict which maps slot names to entities for the provided intent
    """
    slot_name_mapping = dict()
    for utterance in dataset[INTENTS][intent][UTTERANCES]:
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name_mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def get_slot_name_mappings(dataset):
    """Returns a dict which maps intents to their slot name mapping"""
    return {intent: get_slot_name_mapping(dataset, intent)
            for intent in dataset[INTENTS]}
