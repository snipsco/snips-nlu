from __future__ import division, unicode_literals

import json
from builtins import str
from collections import Counter
from copy import deepcopy

from future.utils import iteritems, itervalues
from snips_nlu_ontology import get_all_languages

from snips_nlu.constants import (AUTOMATICALLY_EXTENSIBLE, CAPITALIZE, DATA,
                                 ENTITIES, ENTITY, INTENTS, LANGUAGE,
                                 PARSER_THRESHOLD, SLOT_NAME, SYNONYMS, TEXT,
                                 USE_SYNONYMS, UTTERANCES, VALIDATED, VALUE)
from snips_nlu.entity_parser.builtin_entity_parser import (BuiltinEntityParser,
                                                           is_builtin_entity,
                                                           is_gazetteer_entity)
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.string_variations import get_string_variations
from snips_nlu.utils import validate_key, validate_keys, validate_type


def extract_utterance_entities(dataset):
    entities_values = {ent_name: set() for ent_name in dataset[ENTITIES]}

    for intent in itervalues(dataset[INTENTS]):
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if ENTITY in chunk:
                    entities_values[chunk[ENTITY]].add(chunk[TEXT].strip())
    return {k: list(v) for k, v in iteritems(entities_values)}


def extract_intent_entities(dataset, entity_filter=None):
    intent_entities = {intent: set() for intent in dataset[INTENTS]}
    for intent_name, intent_data in iteritems(dataset[INTENTS]):
        for utterance in intent_data[UTTERANCES]:
            for chunk in utterance[DATA]:
                if ENTITY in chunk:
                    if entity_filter and not entity_filter(chunk[ENTITY]):
                        continue
                    intent_entities[intent_name].add(chunk[ENTITY])
    return intent_entities


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

    utterance_entities_values = extract_utterance_entities(dataset)
    builtin_entity_parser = BuiltinEntityParser.build(dataset=dataset)

    for entity_name, entity in iteritems(dataset[ENTITIES]):
        uterrance_entities = utterance_entities_values[entity_name]
        if is_builtin_entity(entity_name):
            dataset[ENTITIES][entity_name] = \
                validate_and_format_builtin_entity(entity, uterrance_entities)
        else:
            dataset[ENTITIES][entity_name] = validate_and_format_custom_entity(
                entity, uterrance_entities, language, builtin_entity_parser)
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
    return "".join(chunk[TEXT] for chunk in chunks)


def has_any_capitalization(entity_utterances, language):
    for utterance in entity_utterances:
        tokens = tokenize_light(utterance, language)
        if any(t.isupper() or t.istitle() for t in tokens):
            return True
    return False


def add_entity_variations(utterances, entity_variations, entity_value):
    utterances[entity_value] = entity_value
    for variation in entity_variations[entity_value]:
        if variation:
            utterances[variation] = entity_value
    return utterances


def _extract_entity_values(entity):
    values = set()
    for ent in entity[DATA]:
        values.add(ent[VALUE])
        if entity[USE_SYNONYMS]:
            values.update(set(ent[SYNONYMS]))
    return values


def validate_and_format_custom_entity(entity, queries_entities, language,
                                      builtin_entity_parser):
    validate_type(entity, dict)

    # TODO: this is here temporarily, only to allow backward compatibility
    if PARSER_THRESHOLD not in entity:
        entity[PARSER_THRESHOLD] = 1.0

    mandatory_keys = [USE_SYNONYMS, AUTOMATICALLY_EXTENSIBLE, DATA,
                      PARSER_THRESHOLD]
    validate_keys(entity, mandatory_keys, object_label="entity")
    validate_type(entity[USE_SYNONYMS], bool)
    validate_type(entity[AUTOMATICALLY_EXTENSIBLE], bool)
    validate_type(entity[DATA], list)
    validate_type(entity[PARSER_THRESHOLD], float)

    formatted_entity = dict()
    formatted_entity[AUTOMATICALLY_EXTENSIBLE] = entity[
        AUTOMATICALLY_EXTENSIBLE]
    formatted_entity[PARSER_THRESHOLD] = entity[PARSER_THRESHOLD]
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

    validated_utterances = dict()
    # Map original values an synonyms
    for data in entity[DATA]:
        ent_value = data[VALUE]
        if not ent_value:
            continue
        validated_utterances[ent_value] = ent_value
        if use_synonyms:
            for s in data[SYNONYMS]:
                if s and s not in validated_utterances:
                    validated_utterances[s] = ent_value

    # Add variations if not colliding
    all_original_values = _extract_entity_values(entity)
    variations = dict()
    for data in entity[DATA]:
        ent_value = data[VALUE]
        values_to_variate = {ent_value}
        if use_synonyms:
            values_to_variate.update(set(data[SYNONYMS]))
        variations[ent_value] = set(
            v for value in values_to_variate
            for v in get_string_variations(value, language,
                                           builtin_entity_parser))
    variation_counter = Counter(
        [v for vars in itervalues(variations) for v in vars])
    non_colliding_variations = {
        value: [
            v for v in variations if
            v not in all_original_values and variation_counter[v] == 1
        ]
        for value, variations in iteritems(variations)
    }

    for entry in entity[DATA]:
        entry_value = entry[VALUE]
        validated_utterances = add_entity_variations(
            validated_utterances, non_colliding_variations, entry_value)

    # Merge queries entities
    queries_entities_variations = {
        ent: get_string_variations(ent, language, builtin_entity_parser)
        for ent in queries_entities
    }
    for original_ent, variations in iteritems(queries_entities_variations):
        if not original_ent or original_ent in validated_utterances:
            continue
        validated_utterances[original_ent] = original_ent
        for variation in variations:
            if variation and variation not in validated_utterances:
                validated_utterances[variation] = original_ent
    formatted_entity[UTTERANCES] = validated_utterances
    return formatted_entity


def validate_and_format_builtin_entity(entity, queries_entities):
    validate_type(entity, dict)
    return {UTTERANCES: set(queries_entities)}


def get_dataset_gazetteer_entities(dataset, intent=None):
    if intent is not None:
        return extract_intent_entities(dataset, is_gazetteer_entity)[intent]
    return {e for e in dataset[ENTITIES] if is_gazetteer_entity(e)}
