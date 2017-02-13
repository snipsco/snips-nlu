import operator
from collections import defaultdict

from custom_intent_parser.entity_extractor.regex_entity_extractor import (
    RegexEntityExtractor)
from custom_intent_parser.intent_parser.intent_parser import IntentParser


class RegexIntentParser(IntentParser):
    def __init__(self, entity_extractor):
        if not isinstance(entity_extractor, RegexEntityExtractor):
            raise ValueError("entity_extractor must be an instance of %s. "
                             "Found %s" % (RegexEntityExtractor.__name__,
                                           type(entity_extractor)))
        self.entity_extractor = entity_extractor
        self._cache = dict()

    def fitted(self):
        return self.entity_extractor.fitted

    def fit(self, dataset):
        self.entity_extractor = self.entity_extractor.fit(dataset)
        return self

    def parse(self, text):
        if text not in self._cache:
            self._extract_entities(text)
        return self._cache[text]

    def get_intent(self, text):
        if text not in self._cache:
            self._extract_entities(text)
        parse = self._cache[text]
        return parse["intent"]

    def extract_entities(self, text, intent=None):
        if text not in self._cache:
            self._extract_entities(text)
        parse = self._cache[text]
        return parse["entities"]

    def _extract_entities(self, text, intent=None):
        self.check_fitted()
        entities = self.entity_extractor.get_entities(text)

        intent_probs = defaultdict(int)
        num_intents = 0.
        for e in entities:
            intent_probs[e["intent"]] += 1.
            num_intents += 1.

        for k, v in intent_probs.keys():
            intent_probs[k] /= num_intents

        top_intent, top_prob = sorted(intent_probs.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)[0]
        result = {
            "entities": entities,
            "intent": {
                "name": top_intent,
                "prob": top_prob,
            }
        }
        self._cache = result
