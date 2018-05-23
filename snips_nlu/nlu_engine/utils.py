from snips_nlu_utils import normalize

from snips_nlu.builtin_entities import get_builtin_entities, is_builtin_entity
from snips_nlu.constants import (
    UTTERANCES, AUTOMATICALLY_EXTENSIBLE, INTENTS, DATA, SLOT_NAME, ENTITY,
    RES_MATCH_RANGE, RES_VALUE, RES_ENTITY, VALUE, ENTITY_KIND)
from snips_nlu.result import custom_slot, builtin_slot


# pylint:disable=redefined-builtin
def resolve_slots(input, slots, dataset_entities, language, scope):
    builtin_entities = get_builtin_entities(input, language, scope)
    resolved_slots = []
    for slot in slots:
        entity_name = slot[RES_ENTITY]
        raw_value = slot[RES_VALUE]
        if is_builtin_entity(entity_name):
            found = False
            for ent in builtin_entities:
                if ent[ENTITY_KIND] == entity_name and \
                        ent[RES_MATCH_RANGE] == slot[RES_MATCH_RANGE]:
                    resolved_slot = builtin_slot(slot, ent[ENTITY])
                    resolved_slots.append(resolved_slot)
                    found = True
                    break
            if not found:
                builtin_matches = get_builtin_entities(raw_value, language,
                                                       scope=[entity_name])
                if builtin_matches:
                    resolved_slot = builtin_slot(slot,
                                                 builtin_matches[0][VALUE])
                    resolved_slots.append(resolved_slot)
        else:  # custom slot
            entity = dataset_entities[entity_name]
            normalized_raw_value = normalize(raw_value)
            if normalized_raw_value in entity[UTTERANCES]:
                resolved_value = entity[UTTERANCES][normalized_raw_value]
            elif entity[AUTOMATICALLY_EXTENSIBLE]:
                resolved_value = raw_value
            else:
                # entity is skipped
                resolved_value = None

            if resolved_value is not None:
                resolved_slots.append(custom_slot(slot, resolved_value))
    return resolved_slots


# pylint:enable=redefined-builtin

def get_intent_slot_name_mapping(dataset, intent):
    slot_name_mapping = dict()
    intent_data = dataset[INTENTS][intent]
    for utterance in intent_data[UTTERANCES]:
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name_mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping
