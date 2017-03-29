from snips_queries.intent_parser import IntentParser as _BuiltinIntentParser

from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.result import IntentClassificationResult


class BuiltinIntentParser(IntentParser):
    def __init__(self, data_path):
        self.parser = _BuiltinIntentParser(data_path)

    def get_intent(self, text, threshold=0.):
        result = self.parser.get_intent(text, threshold=threshold)
        if len(result) == 0:
            return None
        return IntentClassificationResult(intent_name=result[0]["name"],
                                          probability=result[0]["probability"])

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        return self.parser.get_entities(text, intent)

    @classmethod
    def from_dict(cls, obj_dict):
        if "data_path" in obj_dict:
            return cls(data_path=obj_dict["data_path"])
        raise KeyError("Expected obj_dict to have key 'data_path'")

    def to_dict(self):
        raise NotImplementedError
