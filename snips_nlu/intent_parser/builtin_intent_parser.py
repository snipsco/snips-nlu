from snips_queries.intent_parser import IntentParser as _BuiltinIntentParser

from snips_nlu.constants import BUILTIN_PATH, BUILTIN_BINARY, LANGUAGE
from snips_nlu.result import IntentClassificationResult, ParsedSlot


class BuiltinIntentParser:
    def __init__(self, language, data_path=None, data_binary=None):
        if data_path is not None:
            self.parser = _BuiltinIntentParser(language=language.duckling_code,
                                               data_path=data_path)
        elif data_binary is not None:
            self.parser = _BuiltinIntentParser(language=language.duckling_code,
                                               data_binary=data_binary)
        else:
            raise AssertionError("Expected data_path or data_binary to "
                                 "initialize a BuiltinIntentParser object, but"
                                 " received two None arguments")

    def get_intent(self, text, threshold=0.55):
        result = self.parser.get_intent(text, threshold=threshold)
        if len(result) == 0:
            return None
        return IntentClassificationResult(intent_name=result[0]["name"],
                                          probability=result[0]["probability"])

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        builtin_slots = self.parser.get_entities(text, intent)
        nlu_slots = []
        for slot_name, slot_chunks in builtin_slots.iteritems():
            if len(slot_chunks) > 0:
                for slot_chunk in slot_chunks:
                    rng = [slot_chunk['range']['start'],
                           slot_chunk['range']['end']]
                    nlu_slot = ParsedSlot(match_range=rng,
                                          value=slot_chunk['value'],
                                          entity=slot_name,
                                          slot_name=slot_name)
                    nlu_slots.append(nlu_slot)
        return nlu_slots

    @classmethod
    def from_dict(cls, obj_dict):
        language = obj_dict[LANGUAGE]
        if BUILTIN_PATH in obj_dict:
            return cls(language=language, data_path=obj_dict[BUILTIN_PATH])
        if BUILTIN_BINARY in obj_dict:
            return cls(language=language, data_binary=obj_dict[BUILTIN_BINARY])
        raise KeyError("Expected obj_dict to have key '%s' or "
                       "'%s'" % (BUILTIN_PATH, BUILTIN_BINARY))

    def to_dict(self):
        raise NotImplementedError
