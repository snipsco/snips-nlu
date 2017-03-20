import cPickle
from abc import ABCMeta, abstractmethod

from ..dataset import merge_intent_datasets
from ..intent_parser.intent_parser import CUSTOM_PARSER_TYPE, \
    BUILTIN_PARSER_TYPE
from ..intent_parser.regex_intent_parser import RegexIntentParser
from ..result import result


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        pass


class SnipsNLUEngine(NLUEngine):
    def __init__(self, parsers, custom_first=True):
        super(SnipsNLUEngine, self).__init__()
        self.custom_parsers = filter(
            lambda parser: parser.parser_type == CUSTOM_PARSER_TYPE, parsers)
        self.builtin_parsers = filter(
            lambda parser: parser.parser_type == BUILTIN_PARSER_TYPE, parsers)
        self.custom_first = custom_first
        self.fitted = False

    def parse(self, text):
        if self.custom_first:
            first_parsers = self.custom_parsers
            second_parsers = self.builtin_parsers
        else:
            first_parsers = self.builtin_parsers
            second_parsers = self.custom_parsers

        first_parse = self._parse(text, first_parsers)
        if first_parse["intent"] is not None:
            return first_parse
        else:
            return self._parse(text, second_parsers)

    @staticmethod
    def _parse(text, parsers):
        if len(parsers) == 0:
            return result(text)

        best_parser = None
        best_intent = None
        for parser in parsers:
            res = parser.get_intent(text)
            if best_intent is None or res["prob"] > best_intent["prob"]:
                best_intent = res
                best_parser = parser
        entities = best_parser.get_entities(text)
        return result(text, best_intent, entities)

    def fit(self):
        if self.fitted:
            return
        for parser in self.custom_parsers:
            parser.fit()
        self.fitted = True

    def save_to_pickle_string(self):
        return cPickle.dumps(self)

    @classmethod
    def load_from_dict(cls, obj_dict):
        custom_intent_datasets = obj_dict["custom_intents"]
        merged_dataset = merge_intent_datasets(custom_intent_datasets)
        custom_parsers = [
            RegexIntentParser.load(intent_dataset["name"], merged_dataset) for
            intent_dataset in custom_intent_datasets]
        return SnipsNLUEngine(custom_parsers)

    @classmethod
    def load_from_pickle_string(cls, pkl_str):
        return cPickle.loads(pkl_str)

