from __future__ import unicode_literals

import re
from copy import deepcopy

from nlu_utils import normalize
from semantic_version import Version

from snips_nlu.builtin_entities import BuiltInEntity, is_builtin_entity
from snips_nlu.constants import (TEXT, USE_SYNONYMS, SYNONYMS, DATA, INTENTS,
                                 ENTITIES, ENTITY, SLOT_NAME, UTTERANCES,
                                 LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE,
                                 ENGINE_TYPE, SNIPS_NLU_VERSION, CAPITALIZE)
from snips_nlu.languages import Language
from snips_nlu.tokenization import tokenize_light
from utils import validate_type, validate_key, validate_keys

INTENT_NAME_REGEX = re.compile(r"^[\w\s-]+$")


def extract_queries_entities(dataset):
    entities_values = {ent_name: [] for ent_name in dataset[ENTITIES]}

    for intent in dataset[INTENTS].values():
        for query in intent[UTTERANCES]:
            for chunk in query[DATA]:
                if ENTITY in chunk and not is_builtin_entity(chunk[ENTITY]):
                    entities_values[chunk[ENTITY]].append(chunk[TEXT])
    return {k: list(v) for k, v in entities_values.iteritems()}


def validate_and_format_dataset(dataset, capitalization_threshold=.1):
    dataset = deepcopy(dataset)
    validate_type(dataset, dict)
    mandatory_keys = [INTENTS, ENTITIES, LANGUAGE, SNIPS_NLU_VERSION]
    for key in mandatory_keys:
        validate_key(dataset, key, object_label="dataset")
    Version(dataset[SNIPS_NLU_VERSION])  # Check that the version is semantic
    validate_type(dataset[ENTITIES], dict)
    validate_type(dataset[INTENTS], dict)
    validate_type(dataset[LANGUAGE], basestring)
    language = dataset[LANGUAGE]

    for intent_name, intent in dataset[INTENTS].iteritems():
        validate_intent_name(intent_name)
        validate_and_format_intent(intent, dataset[ENTITIES])

    queries_entities_values = extract_queries_entities(dataset)

    for entity_name, entity in dataset[ENTITIES].iteritems():
        if is_builtin_entity(entity_name):
            dataset[ENTITIES][entity_name] = \
                validate_and_format_builtin_entity(entity)
        else:
            dataset[ENTITIES][entity_name] = validate_and_format_custom_entity(
                entity, queries_entities_values[entity_name], language,
                capitalization_threshold)

    validate_language(dataset[LANGUAGE])
    return dataset


def validate_intent_name(name):
    if not INTENT_NAME_REGEX.match(name):
        raise AssertionError("%s is an invalid intent name. Intent names must "
                             "only use: [a-zA-Z0-9_- ]" % name)


def validate_and_format_intent(intent, entities):
    validate_type(intent, dict)
    validate_key(intent, ENGINE_TYPE, object_label="intent dict")
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
    return intent


def get_text_from_chunks(chunks):
    return ''.join(chunk[TEXT] for chunk in chunks)


def capitalization_ratio(entity_utterances, language):
    capitalizations = []
    for utterance in entity_utterances:
        tokens = tokenize_light(utterance, language)
        for t in tokens:
            if t.isupper() or t.istitle():
                capitalizations.append(1.0)
            else:
                capitalizations.append(0.0)
    if len(capitalizations) == 0:
        return 0
    return sum(capitalizations) / float(len(capitalizations))


def validate_and_format_custom_entity(entity, queries_entities, language,
                                      capitalization_threshold):
    validate_type(entity, dict)
    mandatory_keys = [USE_SYNONYMS, AUTOMATICALLY_EXTENSIBLE, DATA]
    validate_keys(entity, mandatory_keys, object_label="entity")
    validate_type(entity[USE_SYNONYMS], bool)
    validate_type(entity[AUTOMATICALLY_EXTENSIBLE], bool)
    validate_type(entity[DATA], list)

    # Validate format and filter out unused data
    valid_entity_data = []
    for entry in entity[DATA]:
        validate_type(entry, dict)
        validate_keys(entry, [VALUE, SYNONYMS], object_label="entity entry")
        entry[VALUE] = entry[VALUE].strip()
        if len(entry[VALUE]) == 0:
            continue
        validate_type(entry[SYNONYMS], list)
        entry[SYNONYMS] = [s.strip() for s in entry[SYNONYMS]
                           if len(s.strip()) > 0]
        valid_entity_data.append(entry)
    entity[DATA] = valid_entity_data

    # Compute capitalization before normalizing
    # Normalization lowercase and hence lead to bad capitalization calculation
    if entity[USE_SYNONYMS]:
        entities = [s for entry in entity[DATA]
                    for s in entry[SYNONYMS] + [entry[VALUE]]]
    else:
        entities = [entry[VALUE] for entry in entity[DATA]]
    ratio = capitalization_ratio(entities + queries_entities, language)
    entity[CAPITALIZE] = ratio > capitalization_threshold

    # Normalize
    normalize_data = []
    for entry in entity[DATA]:
        normalized_value = normalize(entry[VALUE])
        entry[SYNONYMS] = set(normalize(s) for s in entry[SYNONYMS])
        entry[SYNONYMS].add(normalized_value)
        entry[SYNONYMS] = list(entry[SYNONYMS])
        normalize_data.append(entry)
    entity[DATA] = normalize_data

    # Merge queries_entities
    for value in queries_entities:
        add_entity_value_if_missing(value, entity)

    return entity


def validate_and_format_builtin_entity(entity):
    validate_type(entity, dict)
    return entity


def validate_language(language):
    if language not in Language.language_by_iso_code:
        raise ValueError("Language name must be ISO 639-1,"
                         " found '%s'" % language)


def filter_dataset(dataset, engine_type=None, min_utterances=0):
    """
    Return a deepcopy of the dataset filtered according to parameters
    :param dataset: dataset to filter
    :param engine_type: if not None, only keep intens of type `engine_type`  
    :param min_utterances: keep intents having at least `min_utterances` 
    """
    _dataset = deepcopy(dataset)
    for intent_name, intent in dataset[INTENTS].iteritems():
        if engine_type is not None and intent[ENGINE_TYPE] != engine_type:
            _dataset[INTENTS].pop(intent_name)
        elif len(intent[UTTERANCES]) < min_utterances:
            _dataset[INTENTS].pop(intent_name)
    return _dataset


def add_entity_value_if_missing(value, entity):
    normalized_value = normalize(value)
    if len(normalized_value) == 0:
        return
    if entity[USE_SYNONYMS]:
        entity_values = set(v for entry in entity[DATA]
                            for v in entry[SYNONYMS])
    else:
        entity_values = set(entry[VALUE] for entry in entity[DATA])
    if normalized_value in entity_values:
        return
    entity[DATA].append({VALUE: value, SYNONYMS: [normalized_value]})
