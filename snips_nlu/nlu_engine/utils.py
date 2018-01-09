from snips_nlu.constants import (
    UTTERANCES, AUTOMATICALLY_EXTENSIBLE, INTENTS, DATA, SLOT_NAME, ENTITY,
    RES_MATCH_RANGE, RES_INTENT_NAME, RES_VALUE, RES_ENTITY)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.result import (parsing_result, empty_result,
                              intent_classification_result)
from snips_nlu.utils import ranges_overlap


def parse(text, entities, parsers, intent=None):
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

        valid_slots = []
        slots = parser.get_slots(text, intent_name)
        for s in slots:
            slot_value = s[RES_VALUE]
            # Check if the entity is from a custom intent
            if s[RES_ENTITY] in entities:
                entity = entities[s[RES_ENTITY]]
                if s[RES_VALUE] in entity[UTTERANCES]:
                    slot_value = entity[UTTERANCES][s[RES_VALUE]]
                elif not entity[AUTOMATICALLY_EXTENSIBLE]:
                    continue
            s[RES_VALUE] = slot_value
            valid_slots.append(s)
        return parsing_result(text, intent=res, slots=valid_slots)
    return result


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
