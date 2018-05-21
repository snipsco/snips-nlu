from __future__ import division
from __future__ import unicode_literals

import json
from builtins import str
from copy import deepcopy

from future.utils import itervalues, iteritems
from snips_nlu_ontology import get_all_languages

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (TEXT, USE_SYNONYMS, SYNONYMS, DATA, INTENTS,
                                 ENTITIES, ENTITY, SLOT_NAME, UTTERANCES,
                                 LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE,
                                 CAPITALIZE, VALIDATED)
from snips_nlu.string_variations import get_string_variations
from snips_nlu.tokenization import tokenize_light
from snips_nlu.utils import validate_type, validate_key, validate_keys


def extract_queries_entities(dataset):
    entities_values = {ent_name: set() for ent_name in dataset[ENTITIES]}

    for intent in itervalues(dataset[INTENTS]):
        for query in intent[UTTERANCES]:
            for chunk in query[DATA]:
                if ENTITY in chunk:
                    entities_values[chunk[ENTITY]].add(chunk[TEXT].strip())
    return {k: list(v) for k, v in iteritems(entities_values)}


def validate_and_format_dataset(dataset):
    """Checks that the dataset is valid and format it"""
    # Make this function idempotent
    if dataset.get(VALIDATED, False):
        return dataset
    dataset = deepcopy(dataset)
    dataset = json.loads(json.dumps(dataset))
    validate_type(dataset, dict)
    mandatory_keys = [INTENTS, ENTITIES, LANGUAGE]
    for key in mandatory_keys:
        validate_key(dataset, key, object_label="dataset")
    validate_type(dataset[ENTITIES], dict)
    validate_type(dataset[INTENTS], dict)
    language = dataset[LANGUAGE]
    validate_type(language, str)
    if language not in get_all_languages():
        raise ValueError("Unknown language: '%s'" % language)

    for intent in itervalues(dataset[INTENTS]):
        validate_and_format_intent(intent, dataset[ENTITIES])

    queries_entities_values = extract_queries_entities(dataset)

    for entity_name, entity in iteritems(dataset[ENTITIES]):
        queries_entities = queries_entities_values[entity_name]
        if is_builtin_entity(entity_name):
            dataset[ENTITIES][entity_name] = \
                validate_and_format_builtin_entity(entity, queries_entities)
        else:
            dataset[ENTITIES][entity_name] = validate_and_format_custom_entity(
                entity, queries_entities, language)
    dataset[VALIDATED] = True
    return dataset


def validate_and_format_intent(intent, entities):
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
                if is_builtin_entity(chunk[ENTITY]):
                    continue
                else:
                    validate_key(entities, chunk[ENTITY],
                                 object_label=ENTITIES)
    return intent


def get_text_from_chunks(chunks):
    return ''.join(chunk[TEXT] for chunk in chunks)


def has_any_capitalization(entity_utterances, language):
    for utterance in entity_utterances:
        tokens = tokenize_light(utterance, language)
        if any(t.isupper() or t.istitle() for t in tokens):
            return True
    return False


def add_variation_if_needed(utterances, variation, utterance, language):
    if not variation:
        return utterances
    all_variations = get_string_variations(variation, language)
    for v in all_variations:
        if v not in utterances:
            utterances[v] = utterance
    return utterances


def validate_and_format_custom_entity(entity, queries_entities, language):
    validate_type(entity, dict)
    mandatory_keys = [USE_SYNONYMS, AUTOMATICALLY_EXTENSIBLE, DATA]
    validate_keys(entity, mandatory_keys, object_label="entity")
    validate_type(entity[USE_SYNONYMS], bool)
    validate_type(entity[AUTOMATICALLY_EXTENSIBLE], bool)
    validate_type(entity[DATA], list)

    formatted_entity = dict()
    formatted_entity[AUTOMATICALLY_EXTENSIBLE] = entity[
        AUTOMATICALLY_EXTENSIBLE]
    use_synonyms = entity[USE_SYNONYMS]

    # Validate format and filter out unused data
    valid_entity_data = []
    for entry in entity[DATA]:
        validate_type(entry, dict)
        validate_keys(entry, [VALUE, SYNONYMS], object_label="entity entry")
        entry[VALUE] = entry[VALUE].strip()
        if not entry[VALUE]:
            continue
        validate_type(entry[SYNONYMS], list)
        entry[SYNONYMS] = [s.strip() for s in entry[SYNONYMS]
                           if len(s.strip()) > 0]
        valid_entity_data.append(entry)
    entity[DATA] = valid_entity_data

    # Compute capitalization before normalizing
    # Normalization lowercase and hence lead to bad capitalization calculation
    formatted_entity[CAPITALIZE] = has_any_capitalization(queries_entities,
                                                          language)

    # Normalize
    validated_data = dict()
    for entry in entity[DATA]:
        entry_value = entry[VALUE]
        validated_data = add_variation_if_needed(
            validated_data, entry_value, entry_value, language)

        if use_synonyms:
            for s in entry[SYNONYMS]:
                validated_data = add_variation_if_needed(
                    validated_data, s, entry_value, language)

    formatted_entity[UTTERANCES] = validated_data
    # Merge queries_entities
    for value in queries_entities:
        formatted_entity = add_entity_value_if_missing(
            value, formatted_entity, language)

    return formatted_entity


def validate_and_format_builtin_entity(entity, queries_entities):
    validate_type(entity, dict)
    return {UTTERANCES: set(queries_entities)}


def add_entity_value_if_missing(value, entity, language):
    entity[UTTERANCES] = add_variation_if_needed(entity[UTTERANCES], value,
                                                 value, language)
    return entity
