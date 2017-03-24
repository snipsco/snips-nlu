from collections import namedtuple

_IntentClassificationResult = namedtuple('_IntentClassificationResult',
                                         'intent_name probability')
_ParsedSlot = namedtuple('_ParsedSlot', 'match_range value entity slot_name')
_Result = namedtuple('_Result', 'text parsed_intent parsed_slots')


class IntentClassificationResult(_IntentClassificationResult):
    def __init__(self, intent_name, probability):
        super(IntentClassificationResult, self).__init__([intent_name,
                                                          probability])

    def as_dict(self):
        return {
            "intent_name": self.intent_name,
            "probability": self.probability
        }


class ParsedSlot(_ParsedSlot):
    def __init__(self, match_range, value, entity, slot_name):
        super(ParsedSlot, self).__init__(
            [match_range, value, entity, slot_name])

    def as_dict(self):
        return {
            "match_range": [self.match_range[0], self.match_range[1]],
            "value": self.value,
            "slot_name": self.slot_name
        }


class Result(_Result):
    def __init__(self, text, parsed_intent, parsed_slots):
        super(Result, self).__init__([text, parsed_intent, parsed_slots])

    def as_dict(self):
        if self.parsed_intent is not None:
            parsed_intent = self.parsed_intent.as_dict()
        else:
            parsed_intent = None
        if self.parsed_slots is not None:
            parsed_entities = map(lambda slot: slot.as_dict(),
                                  self.parsed_slots)
        else:
            parsed_entities = None
        return {
            "text": self.text,
            "parsed_intent": parsed_intent,
            "parsed_slots": parsed_entities
        }
