from snips_nlu.builtin_entities import (
    get_builtin_entities, is_builtin_entity, BuiltInEntity)
from snips_nlu.constants import (
    UTTERANCES, AUTOMATICALLY_EXTENSIBLE, INTENTS, DATA, SLOT_NAME, ENTITY,
    RES_MATCH_RANGE, RES_INTENT_NAME, RES_VALUE, RES_ENTITY, VALUE)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.result import (parsing_result, empty_result,
                              intent_classification_result, custom_slot,
                              builtin_slot)
from snips_nlu.utils import ranges_overlap


def parse(text, entities, language, parsers, intent=None):
    if not parsers:
        return empty_result(text)

    result = empty_result(text) if intent is None else parsing_result(
        text, intent=intent_classification_result(intent, 1.0), slots=[])

    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue

        intent_name = res[RES_INTENT_NAME]
        if intent is not None:
            if intent_name != intent:
                continue
            res = intent_classification_result(intent_name, 1.0)

        slots = parser.get_slots(text, intent_name)
        scope = [BuiltInEntity.from_label(s[RES_ENTITY]) for s in slots
                 if is_builtin_entity(s[RES_ENTITY])]
        resolved_slots = resolve_slots(text, slots, entities, language, scope)
        return parsing_result(text, intent=res, slots=resolved_slots)
    return result


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
                if ent[ENTITY].label == entity_name and \
                        ent[RES_MATCH_RANGE] == slot[RES_MATCH_RANGE]:
                    resolved_slot = builtin_slot(slot, ent[VALUE])
                    resolved_slots.append(resolved_slot)
                    found = True
                    break
            if not found:
                builtin_entity = BuiltInEntity.from_label(entity_name)
                builtin_matches = get_builtin_entities(raw_value, language,
                                                       scope=[builtin_entity])
                if builtin_matches:
                    resolved_slot = builtin_slot(slot,
                                                 builtin_matches[0][VALUE])
                    resolved_slots.append(resolved_slot)
        else:  # custom slot
            entity = dataset_entities[entity_name]
            if raw_value in entity[UTTERANCES]:
                resolved_value = entity[UTTERANCES][raw_value]
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


def enrich_slots(slots, other_slots):
    enriched_slots = list(slots)
    for slot in other_slots:
        if any(ranges_overlap(slot[RES_MATCH_RANGE], s[RES_MATCH_RANGE])
               for s in enriched_slots):
            continue
        enriched_slots.append(slot)
    return enriched_slots


def get_fitted_slot_filler(engine, dataset, intent):
    dataset = validate_and_format_dataset(dataset)
    probabilistic_parser = _get_probabilistic_intent_parser(engine)
    return probabilistic_parser.get_fitted_slot_filler(dataset, intent)


def add_fitted_slot_filler(engine, intent, model_data):
    probabilistic_parser = _get_probabilistic_intent_parser(engine)
    probabilistic_parser.add_fitted_slot_filler(intent, model_data)


def _get_probabilistic_intent_parser(engine):
    probabilistic_parser = None
    for intent_parser in engine.intent_parsers:
        if intent_parser.unit_name == ProbabilisticIntentParser.unit_name:
            probabilistic_parser = intent_parser
    if probabilistic_parser is None:
        probabilistic_parser_config = None
        for parser_config in engine.config.intent_parsers_configs:
            if parser_config.unit_name == \
                    ProbabilisticIntentParser.unit_name:
                probabilistic_parser_config = parser_config
                break
        probabilistic_parser = ProbabilisticIntentParser(
            probabilistic_parser_config)
        engine.intent_parsers.append(probabilistic_parser)
    return probabilistic_parser
