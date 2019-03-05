from __future__ import division, unicode_literals

import json
from builtins import str
from collections import Counter
from copy import deepcopy

from future.utils import iteritems, itervalues
from snips_nlu_parsers import get_all_languages

from snips_nlu.constants import (
    AUTOMATICALLY_EXTENSIBLE, CAPITALIZE, DATA, ENTITIES, ENTITY, INTENTS,
    LANGUAGE, MATCHING_STRICTNESS, SLOT_NAME, SYNONYMS, TEXT, USE_SYNONYMS,
    UTTERANCES, VALIDATED, VALUE)
from snips_nlu.dataset import extract_utterance_entities
from snips_nlu.entity_parser.builtin_entity_parser import (
    BuiltinEntityParser, is_builtin_entity)
from snips_nlu.exceptions import DatasetFormatError
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.string_variations import get_string_variations
from snips_nlu.common.dataset_utils import validate_type, validate_key, \
    validate_keys


def validate_and_format_dataset(dataset):
    """Checks that the dataset is valid and format it

    Raise:
        DatasetFormatError: When the dataset format is wrong
    """
    # Make this function idempotent
    if dataset.get(VALIDATED, False):
        return dataset
    dataset = deepcopy(dataset)
    dataset = json.loads(json.dumps(dataset))
    validate_type(dataset, dict, object_label="dataset")
    mandatory_keys = [INTENTS, ENTITIES, LANGUAGE]
    for key in mandatory_keys:
        validate_key(dataset, key, object_label="dataset")
    validate_type(dataset[ENTITIES], dict, object_label="entities")
    validate_type(dataset[INTENTS], dict, object_label="intents")
    language = dataset[LANGUAGE]
    validate_type(language, str, object_label="language")
    if language not in get_all_languages():
        raise DatasetFormatError("Unknown language: '%s'" % language)

    for intent in itervalues(dataset[INTENTS]):
        _validate_and_format_intent(intent, dataset[ENTITIES])

    utterance_entities_values = extract_utterance_entities(dataset)
    builtin_entity_parser = BuiltinEntityParser.build(dataset=dataset)

    for entity_name, entity in iteritems(dataset[ENTITIES]):
        uterrance_entities = utterance_entities_values[entity_name]
        if is_builtin_entity(entity_name):
            dataset[ENTITIES][entity_name] = \
                _validate_and_format_builtin_entity(entity, uterrance_entities)
        else:
            dataset[ENTITIES][entity_name] = \
                _validate_and_format_custom_entity(
                    entity, uterrance_entities, language,
                    builtin_entity_parser)
    dataset[VALIDATED] = True
    return dataset


def _validate_and_format_intent(intent, entities):
    validate_type(intent, dict, "intent")
    validate_key(intent, UTTERANCES, object_label="intent dict")
    validate_type(intent[UTTERANCES], list, object_label="utterances")
    for utterance in intent[UTTERANCES]:
        validate_type(utterance, dict, object_label="utterance")
        validate_key(utterance, DATA, object_label="utterance")
        validate_type(utterance[DATA], list, object_label="utterance data")
        for chunk in utterance[DATA]:
            validate_type(chunk, dict, object_label="utterance chunk")
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


def _has_any_capitalization(entity_utterances, language):
    for utterance in entity_utterances:
        tokens = tokenize_light(utterance, language)
        if any(t.isupper() or t.istitle() for t in tokens):
            return True
    return False


def _add_entity_variations(utterances, entity_variations, entity_value):
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


def _validate_and_format_custom_entity(entity, queries_entities, language,
                                       builtin_entity_parser):
    validate_type(entity, dict, object_label="entity")

    # TODO: this is here temporarily, only to allow backward compatibility
    if MATCHING_STRICTNESS not in entity:
        strictness = entity.get("parser_threshold", 1.0)

        entity[MATCHING_STRICTNESS] = strictness

    mandatory_keys = [USE_SYNONYMS, AUTOMATICALLY_EXTENSIBLE, DATA,
                      MATCHING_STRICTNESS]
    validate_keys(entity, mandatory_keys, object_label="custom entity")
    validate_type(entity[USE_SYNONYMS], bool, object_label="use_synonyms")
    validate_type(entity[AUTOMATICALLY_EXTENSIBLE], bool,
                  object_label="automatically_extensible")
    validate_type(entity[DATA], list, object_label="entity data")
    validate_type(entity[MATCHING_STRICTNESS], (float, int),
                  object_label="matching_strictness")

    formatted_entity = dict()
    formatted_entity[AUTOMATICALLY_EXTENSIBLE] = entity[
        AUTOMATICALLY_EXTENSIBLE]
    formatted_entity[MATCHING_STRICTNESS] = entity[MATCHING_STRICTNESS]
    use_synonyms = entity[USE_SYNONYMS]

    # Validate format and filter out unused data
    valid_entity_data = []
    for entry in entity[DATA]:
        validate_type(entry, dict, object_label="entity entry")
        validate_keys(entry, [VALUE, SYNONYMS], object_label="entity entry")
        entry[VALUE] = entry[VALUE].strip()
        if not entry[VALUE]:
            continue
        validate_type(entry[SYNONYMS], list, object_label="entity synonyms")
        entry[SYNONYMS] = [s.strip() for s in entry[SYNONYMS]
                           if len(s.strip()) > 0]
        valid_entity_data.append(entry)
    entity[DATA] = valid_entity_data

    # Compute capitalization before normalizing
    # Normalization lowercase and hence lead to bad capitalization calculation
    formatted_entity[CAPITALIZE] = _has_any_capitalization(queries_entities,
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
        validated_utterances = _add_entity_variations(
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


def _validate_and_format_builtin_entity(entity, queries_entities):
    validate_type(entity, dict, object_label="builtin entity")
    return {UTTERANCES: set(queries_entities)}
