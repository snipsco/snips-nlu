import re

from snips_nlu.built_in_entities import BuiltInEntity
from utils import validate_type, validate_key, validate_keys

INTENT_NAME_REGEX = re.compile(r"^[\w\s-]+$")
ENTITY_NAME_REGEX = re.compile("^[\w]+$")


def validate_dataset(dataset):
    validate_type(dataset, dict)
    mandatory_keys = ["intents", "entities"]
    for key in mandatory_keys:
        validate_key(dataset, key, object_label="dataset")
    validate_type(dataset["entities"], dict)
    validate_type(dataset["intents"], dict)
    entities = set()
    for entity_name, entity in dataset["entities"].iteritems():
        validate_entity_name(entity_name)
        entities.add(entity_name)
        validate_entity(entity)
    for intent_name, intent in dataset["intents"].iteritems():
        validate_intent_name(intent_name)
        validate_intent(intent, entities)


def validate_intent_name(name):
    if not INTENT_NAME_REGEX.match(name):
        raise AssertionError("%s is an invalid intent name. Intent names must "
                             "only use: [a-zA-Z0-9_- ]" % name)


def validate_intent(intent, entities):
    validate_type(intent, dict)
    validate_key(intent, "utterances", object_label="intent dict")
    validate_type(intent["utterances"], list)
    for utterance in intent["utterances"]:
        validate_type(utterance, dict)
        validate_key(utterance, "data", object_label="utterance")
        validate_type(utterance["data"], list)
        for chunk in utterance["data"]:
            validate_type(chunk, dict)
            validate_key(chunk, "text", object_label="chunk")
            if len(chunk.keys()) > 1:
                mandatory_keys = ["entity", "slot_name"]
                validate_keys(chunk, mandatory_keys, object_label="chunk")
                if chunk["entity"] in BuiltInEntity.built_in_entity_by_label:
                    continue
                else:
                    validate_key(entities, chunk["entity"],
                                 object_label="entities")


def validate_entity_name(name):
    if not ENTITY_NAME_REGEX.match(name):
        raise ValueError("Entity name must only contain [0-9a-zA-Z_],"
                         " found '%s'" % name)


def validate_entity(entity):
    validate_type(entity, dict)
    mandatory_keys = ["use_synonyms", "automatically_extensible", "data"]
    validate_keys(entity, mandatory_keys, object_label="entity")
    validate_type(entity["use_synonyms"], bool)
    validate_type(entity["automatically_extensible"], bool)
    validate_type(entity["data"], list)
    for entry in entity["data"]:
        validate_type(entry, dict)
        validate_keys(entry, ["value", "synonyms"],
                      object_label="entity entry")
        validate_type(entry["synonyms"], list)
