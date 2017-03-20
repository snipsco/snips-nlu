from collections import namedtuple

IntentClassificationResult = namedtuple('IntentClassificationResult',
                                        'intent_name probability')
ParsedEntity = namedtuple('ParsedEntity', 'match_range value entity slot_name')
Result = namedtuple('Result', 'text parsed_intent parsed_entities')


def intent_classification_result(intent_name, probability):
    return IntentClassificationResult(intent_name, probability)


def parsed_entity(match_range, value, entity, slot_name):
    return ParsedEntity(match_range, value, entity, slot_name)


def result(text, parsed_intent=None, parsed_entities=None):
    return Result(text=text, parsed_intent=parsed_intent,
                  parsed_entities=parsed_entities)
