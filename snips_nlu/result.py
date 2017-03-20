from collections import namedtuple

IntentClassificationResult = namedtuple('IntentClassificationResult',
                                        'intent_name probability')
ParsedEntity = namedtuple('ParsedEntity', 'match_range value entity slot_name')
Result = namedtuple('Result', 'text parsed_intent parsed_entities')
