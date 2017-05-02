from copy import copy
from itertools import groupby, permutations

from snips_nlu.built_in_entities import BuiltInEntity
from snips_nlu.constants import ENTITY, MATCH_RANGE, INTENTS, UTTERANCES, DATA, \
    SLOT_NAME, ENTITIES, USE_SYNONYMS, AUTOMATICALLY_EXTENSIBLE, SYNONYMS, \
    VALUE
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_utils import positive_tagging, tags_to_slots


def augment_slots(text, tokens, tags, intent_slots_mapping, builtin_entities,
                  missing_slots, tagger):
    augmented_tags = copy(tags)
    grouped_entities = groupby(builtin_entities, key=lambda s: s[ENTITY])
    for entity, spans in grouped_entities:
        spans_ranges = [span[MATCH_RANGE] for span in spans]
        tokens_indexes = spans_to_tokens_indexes(spans_ranges, tokens)
        related_slots = set(s for s in missing_slots
                            if intent_slots_mapping[s] == entity.label)
        slots_permutations = permutations(related_slots)
        best_updated_tags = augmented_tags
        best_permutation_score = -1
        for slots in slots_permutations:
            updated_tags = copy(augmented_tags)
            for slot_index, slot in enumerate(slots):
                if slot_index >= len(tokens_indexes):
                    break
                indexes = tokens_indexes[slot_index]
                sub_tags_sequence = positive_tagging(tagger.tagging_scheme,
                                                     slot, len(indexes))
                updated_tags[indexes[0]:indexes[-1] + 1] = sub_tags_sequence
            score = tagger.get_sequence_probability(tokens, updated_tags)
            if score > best_permutation_score:
                best_updated_tags = updated_tags
                best_permutation_score = score
        augmented_tags = best_updated_tags
    return tags_to_slots(text, tokens, augmented_tags, tagger.tagging_scheme,
                         intent_slots_mapping)


def spans_to_tokens_indexes(spans, tokens):
    tokens_indexes = []
    for span_start, span_end in spans:
        indexes = []
        for i, token in enumerate(tokens):
            if span_end > token.start and span_start < token.end:
                indexes.append(i)
        tokens_indexes.append(indexes)
    return tokens_indexes


def get_slot_name_mapping(dataset):
    """
    Returns a dict which maps slot names to entities
    """
    slot_name_mapping = dict()
    for intent_name, intent in dataset[INTENTS].iteritems():
        _dict = dict()
        slot_name_mapping[intent_name] = _dict
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    _dict[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def get_intent_custom_entities(dataset, intent):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent][UTTERANCES]:
        for c in utterance[DATA]:
            if ENTITY in c:
                intent_entities.add(c[ENTITY])
    custom_entities = dict()
    for ent in intent_entities:
        if ent not in BuiltInEntity.built_in_entity_by_label:
            custom_entities[ent] = dataset[ENTITIES][ent]
    return custom_entities


def snips_nlu_entities(dataset):
    entities = dict()
    for entity_name, entity in dataset[ENTITIES].iteritems():
        entity_data = dict()
        use_synonyms = entity[USE_SYNONYMS]
        automatically_extensible = entity[AUTOMATICALLY_EXTENSIBLE]
        entity_data[AUTOMATICALLY_EXTENSIBLE] = automatically_extensible

        entity_utterances = dict()
        for data in entity[DATA]:
            if use_synonyms:
                for s in data[SYNONYMS]:
                    entity_utterances[s] = data[VALUE]
            else:
                entity_utterances[data[VALUE]] = data[VALUE]
        entity_data[UTTERANCES] = entity_utterances
        entities[entity_name] = entity_data
    return entities


def empty_result(text):
    return Result(text=text, parsed_intent=None, parsed_slots=None)
