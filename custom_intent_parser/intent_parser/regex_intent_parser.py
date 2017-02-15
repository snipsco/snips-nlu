import operator
from collections import defaultdict

from custom_intent_parser.entity_extractor.regex_entity_extractor import (
    RegexEntityExtractor)
from custom_intent_parser.intent_parser.intent_parser import IntentParser
from custom_intent_parser.result import result, intent_classification_result
from custom_intent_parser.utils import LimitedSizeDict


class RegexIntentParser(IntentParser):
    def __init__(self, entity_extractor, cache_size=100):
        if not isinstance(entity_extractor, RegexEntityExtractor):
            raise ValueError("entity_extractor must be an instance of %s. "
                             "Found %s" % (RegexEntityExtractor.__name__,
                                           type(entity_extractor)))
        self.entity_extractor = entity_extractor
        self._cache = LimitedSizeDict(size_limit=cache_size)

    def fitted(self):
        return self.entity_extractor.fitted

    def fit(self, dataset):
        self.entity_extractor = self.entity_extractor.fit(dataset)
        return self

    def parse(self, text):
        if text not in self._cache:
            self._update_cache(text)
        return self._cache[text]

    def get_intent(self, text):
        if text not in self._cache:
            self._update_cache(text)
        parse = self._cache[text]
        return intent_classification_result(
            parse["intent"].get("name"), parse["intent"].get("prob"))

    def get_entities(self, text, intent=None):
        if text not in self._cache:
            self._update_cache(text)
        parse = self._cache[text]
        return parse["entities"]

    def _update_cache(self, text):
        self.check_fitted()
        entities = self.entity_extractor.get_entities(text)

        if len(entities) == 0:
            res = result(text)
            self._cache[text] = res
            return

        intent_probs = defaultdict(int)
        num_intents = 0.
        for e in entities:
            intent_probs[e["intent"]] += 1.
            num_intents += 1.

        for k, v in intent_probs.iteritems():
            intent_probs[k] /= num_intents

        top_intent, top_prob = sorted(intent_probs.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)[0]
        intent = intent_classification_result(top_intent, top_prob)
        for e in entities:
            e.pop("intent", None)
        res = result(text, intent, entities)
        self._cache[text] = res
