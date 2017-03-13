import io
import json
import operator
import os
from collections import defaultdict

from snips_nlu.built_in_intents import (
    get_built_in_intents, get_built_in_intent_entities, BuiltInIntent)
from snips_nlu.entity_extractor.regex_entity_extractor import (
    RegexEntityExtractor)
from snips_nlu.nlu_engine.nlu_engine import SnipsNLUEngine
from snips_nlu.result import result, intent_classification_result
from snips_nlu.utils import LimitedSizeDict


class RegexNLUEngine(SnipsNLUEngine):
    def __init__(self, entity_extractor, built_in_intents=None, cache=None,
                 cache_size=100):
        super(RegexNLUEngine, self).__init__()
        if not isinstance(entity_extractor, RegexEntityExtractor):
            raise ValueError("entity_extractor must be an instance of %s. "
                             "Found %s" % (RegexEntityExtractor.__name__,
                                           type(entity_extractor)))
        if built_in_intents is None:
            built_in_intents = []
        self.built_in_intents = built_in_intents

        self.entity_extractor = entity_extractor
        if cache is None:
            cache = LimitedSizeDict(size_limit=cache_size)
        self._cache = cache

    def fit(self, dataset):
        self._cache.clear()
        self.entity_extractor = self.entity_extractor.fit(dataset)
        self.fitted = True
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
            if len(self.built_in_intents) > 0:
                intents = get_built_in_intents(text, self.built_in_intents)
            else:
                intents = []
            if len(intents) == 0:
                res = result(text)
            else:
                intent = max(intents, key=lambda x: x["prob"])
                entities = get_built_in_intent_entities(
                    text, BuiltInIntent[intent["intent"]])
                res = result(text, intent, entities)
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

    @classmethod
    def from_dataset(cls, dataset, built_in_intents):
        extractor = RegexEntityExtractor()
        extractor = extractor.fit(dataset)
        return cls(extractor, built_in_intents=built_in_intents)
