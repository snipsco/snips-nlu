import re
from copy import deepcopy

from snips_nlu.built_in_entities import BuiltInEntity
from snips_nlu.constants import (TEXT, USE_SYNONYMS, SYNONYMS, DATA, INTENTS,
                                 ENTITIES, ENTITY, SLOT_NAME, UTTERANCES,
                                 LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE)
from snips_nlu.languages import Language
from utils import validate_type, validate_key, validate_keys

INTENT_NAME_REGEX = re.compile(r"^[\w\s-]+$")
ENTITY_NAME_REGEX = re.compile("^[\w]+$")


def validate_and_format_dataset(dataset):
    dataset = deepcopy(dataset)
    validate_type(dataset, dict)
    mandatory_keys = [INTENTS, ENTITIES, LANGUAGE]
    for key in mandatory_keys:
        validate_key(dataset, key, object_label="dataset")
    validate_type(dataset[ENTITIES], dict)
    validate_type(dataset[INTENTS], dict)
    validate_type(dataset[LANGUAGE], basestring)
    entities = set()
    for entity_name, entity in dataset[ENTITIES].iteritems():
        entities.add(entity_name)
        dataset[ENTITIES][entity_name] = validate_and_format_entity(entity)
    for intent_name, intent in dataset[INTENTS].iteritems():
        validate_intent_name(intent_name)
        validate_intent(intent, entities)
    validate_language(dataset[LANGUAGE])
    return dataset


def validate_intent_name(name):
    if not INTENT_NAME_REGEX.match(name):
        raise AssertionError("%s is an invalid intent name. Intent names must "
                             "only use: [a-zA-Z0-9_- ]" % name)


def validate_intent(intent, entities):
    validate_type(intent, dict)
    validate_key(intent, UTTERANCES, object_label="intent dict")
    validate_type(intent[UTTERANCES], list)
    for utterance in intent[UTTERANCES]:
        validate_type(utterance, dict)
        validate_key(utterance, DATA, object_label="utterance")
        validate_type(utterance[DATA], list)
        for chunk in utterance[DATA]:
            validate_type(chunk, dict)
            validate_key(chunk, TEXT, object_label="chunk")
            if ENTITY in chunk or SLOT_NAME in chunk:
                mandatory_keys = [ENTITY, SLOT_NAME]
                validate_keys(chunk, mandatory_keys, object_label="chunk")
                if chunk[ENTITY] in BuiltInEntity.built_in_entity_by_label:
                    continue
                else:
                    validate_key(entities, chunk[ENTITY],
                                 object_label=ENTITIES)


def validate_and_format_entity(entity):
    validate_type(entity, dict)
    mandatory_keys = [USE_SYNONYMS, AUTOMATICALLY_EXTENSIBLE, DATA]
    validate_keys(entity, mandatory_keys, object_label="entity")
    validate_type(entity[USE_SYNONYMS], bool)
    validate_type(entity[AUTOMATICALLY_EXTENSIBLE], bool)
    validate_type(entity[DATA], list)
    for entry in entity[DATA]:
        validate_type(entry, dict)
        validate_keys(entry, [VALUE, SYNONYMS],
                      object_label="entity entry")
        validate_type(entry[SYNONYMS], list)
        if entry[VALUE] not in entry[SYNONYMS]:
            entry[SYNONYMS].append(entry[VALUE])

    return entity


def validate_language(language):
    if language not in Language.language_by_iso_code:
        raise ValueError("Language name must be ISO 639-1,"
                         " found '%s'" % language)
