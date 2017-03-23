from snips_queries.intent_parser import IntentParser as _BuiltinIntentParser
from ..result import IntentClassificationResult
from ..intent_parser.intent_parser import IntentParser


class BuiltinIntentParser(IntentParser):
    def __init__(self, data_path):
        self.parser = _BuiltinIntentParser(data_path, intents=[])

    def get_intent(self, text, threshold=0.):
        result = self.parser.get_intent(text, threshold=threshold)
        if len(result) == 0:
            return None
        return IntentClassificationResult(intent_name=result[0]["name"],
                                          probability=result[0]["probability"])

    def get_entities(self, text, intent):
        return self.parser.get_entities(text, intent)
