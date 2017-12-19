from snips_nlu.constants import (
    UTTERANCES, AUTOMATICALLY_EXTENSIBLE, INTENTS, DATA, SLOT_NAME, ENTITY)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.result import (
    empty_result, Result, IntentClassificationResult, ParsedSlot)


def parse(text, entities, parsers, intent=None):
    if not parsers:
        return empty_result(text)

    result = empty_result(text) if intent is None else Result(
        text, parsed_intent=IntentClassificationResult(intent, 1.0),
        parsed_slots=[])

    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue

        intent_name = res.intent_name
        if intent is not None:
            if intent_name != intent:
                continue
            res = IntentClassificationResult(intent_name, 1.0)

        valid_slot = []
        slots = parser.get_slots(text, intent_name)
        for s in slots:
            slot_value = s.value
            # Check if the entity is from a custom intent
            if s.entity in entities:
                entity = entities[s.entity]
                if s.value in entity[UTTERANCES]:
                    slot_value = entity[UTTERANCES][s.value]
                elif not entity[AUTOMATICALLY_EXTENSIBLE]:
                    continue
            s = ParsedSlot(s.match_range, slot_value, s.entity,
                           s.slot_name)
            valid_slot.append(s)
        return Result(text, parsed_intent=res, parsed_slots=valid_slot)
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
        if any((slot.match_range[1] > s.match_range[0])
               and (slot.match_range[0] < s.match_range[1])
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
