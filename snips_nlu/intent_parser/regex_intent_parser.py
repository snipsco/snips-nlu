import io
import json
import operator
import os
from collections import defaultdict

from custom_intent_parser import CustomIntentParser
from ..entity_extractor.regex_entity_extractor import RegexEntityExtractor
from ..result import result, intent_classification_result
from ..utils import LimitedSizeDict


class RegexIntentParser(CustomIntentParser):
    def __init__(self, entity_extractor, cache=None, cache_size=100):
        if not isinstance(entity_extractor, RegexEntityExtractor):
            raise ValueError("entity_extractor must be an instance of %s. "
                             "Found %s" % (RegexEntityExtractor.__name__,
                                           type(entity_extractor)))
        super(CustomIntentParser, self).__init__()
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
            self._cache[text] = self._parse(text)
        return self._cache[text]

    def _parse(self, text):
        if not self.fitted:
            raise AssertionError("Custom intent parser must be fitted before "
                                 "calling `parse` method")
        entities = self.entity_extractor.get_entities(text)
        if len(entities) == 0:
            return result(text)
        else:
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
            return result(text, intent, valid_entities)

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
    def from_dataset(cls, dataset):
        extractor = RegexEntityExtractor()
        extractor = extractor.fit(dataset)
        return cls(extractor)
