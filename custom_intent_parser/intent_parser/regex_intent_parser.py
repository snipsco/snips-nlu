import io
import json
import operator
import os
from collections import defaultdict

from custom_intent_parser.entity_extractor.regex_entity_extractor import (
    RegexEntityExtractor)
from custom_intent_parser.intent_parser.intent_parser import IntentParser
from custom_intent_parser.result import result, intent_classification_result
from custom_intent_parser.utils import LimitedSizeDict


class RegexIntentParser(IntentParser):
    def __init__(self, entity_extractor, cache=None, cache_size=100):
        if not isinstance(entity_extractor, RegexEntityExtractor):
            raise ValueError("entity_extractor must be an instance of %s. "
                             "Found %s" % (RegexEntityExtractor.__name__,
                                           type(entity_extractor)))
        self.entity_extractor = entity_extractor
        if cache is None:
            cache = LimitedSizeDict(size_limit=cache_size)
        self._cache = cache

    def fitted(self):
        return self.entity_extractor.fitted

    def fit(self, dataset):
        self._cache.clear()
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
        intent = parse["intent"]
        prob = None
        if intent is not None:
            prob = intent["prob"]
            intent = intent["name"]
        return intent_classification_result(intent, prob)

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
        for e in entities:
            intent_probs[e["intent"]] += 1.

        for k, v in intent_probs.iteritems():
            intent_probs[k] /= float(len(entities))

        top_intent, top_prob = max(intent_probs.items(),
                                   key=operator.itemgetter(1))
        intent = intent_classification_result(top_intent, top_prob)
        # Remove entities from wrong intent and the intent key
        valid_entities = []
        for e in entities:
            if e["intent"] == top_intent:
                e.pop("intent", None)
                valid_entities.append(e)
        res = result(text, intent, valid_entities)
        self._cache[text] = res

    @staticmethod
    def entity_extractor_file_name(path):
        return os.path.join(path, "entity_extractor.json")

    @staticmethod
    def intent_parser_file_name(path):
        return os.path.join(path, "intent_parser.json")

    def save(self, path):
        self_as_dict = dict()
        self_as_dict["cache_size"] = self._cache.size_limit
        self_as_dict["cache_items"] = self._cache.items()
        os.mkdir(path)
        self_as_dict["entity_extractor_path"] = \
            self.entity_extractor_file_name(path)
        with io.open(self.intent_parser_file_name(path), "w",
                     encoding="utf8") as f:
            data = json.dumps(self_as_dict, indent=2)
            f.write(unicode(data))
        self.entity_extractor.save(self.entity_extractor_file_name(path))

    @classmethod
    def load(cls, path):
        with io.open(cls.intent_parser_file_name(path), encoding="utf8") as f:
            data = json.load(f)
        entity_extractor = RegexEntityExtractor.load(
            cls.entity_extractor_file_name(path))
        cache_size = data["cache_size"]
        cache = LimitedSizeDict([(k, v) for k, v in data["cache_items"]],
                                size_limit=cache_size)
        return cls(entity_extractor, cache)
