from intent_parser import IntentParser
from ..built_in_intents import get_built_in_intents, \
    get_built_in_intent_entities, BuiltInIntent
from ..result import result
from ..utils import LimitedSizeDict


class BuiltinIntentParser(IntentParser):
    def __init__(self, builtin_intents, cache=None, cache_size=100):
        self._builtin_intents = None
        self.builtin_intents = builtin_intents
        if cache is None:
            cache = LimitedSizeDict(size_limit=cache_size)
        self._cache = cache

    @property
    def builtin_intents(self):
        return self._builtin_intents

    @builtin_intents.setter
    def builtin_intents(self, value):
        for intent in value:
            if not isinstance(intent, BuiltInIntent):
                raise ValueError("Expected a BuiltInIntent, found: %s"
                                 % type(intent))
        self._builtin_intents = value

    def parse(self, text):
        if text not in self._cache:
            self._cache[text] = self._parse(text)
        return self._cache[text]

    def _parse(self, text):
        if len(self.builtin_intents) == 0:
            return result(text)

        most_likely_intent = self.get_intent(text)
        if most_likely_intent is None:
            return result(text)

        entities = get_built_in_intent_entities(
            text, BuiltInIntent[most_likely_intent["intent"]])
        return result(text, most_likely_intent, entities)

    def get_intent(self, text):
        intents = get_built_in_intents(text, self.builtin_intents)
        if len(intents) == 0:
            return None
        else:
            return max(intents, key=lambda x: x["prob"])

    def get_entities(self, text, intent=None):
        if intent is None:
            most_likely_intent = self.get_intent(text)
            if most_likely_intent is None:
                return []
            builtin_intent = BuiltInIntent[most_likely_intent["intent"]]
        else:
            builtin_intent = BuiltInIntent[intent]

        return get_built_in_intent_entities(text, builtin_intent)
