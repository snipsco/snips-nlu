from collections import namedtuple

IntentClassificationResult = namedtuple('IntentClassificationResult',
                                        'intent_name probability')
ParsedSlot = namedtuple('ParsedSlot', 'match_range value entity slot_name')
Result = namedtuple('Result', 'text parsed_intent parsed_slots')
