import cPickle
from abc import ABCMeta, abstractmethod

from ..intent_parser.intent_parser import CUSTOM_PARSER_TYPE, \
    BUILTIN_PARSER_TYPE
from ..intent_parser.regex_intent_parser import RegexIntentParser
from ..result import Result
from ..dataset import validate_dataset


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        pass


class SnipsNLUEngine(NLUEngine):
    def __init__(self, parsers=None, custom_first=True):
        super(SnipsNLUEngine, self).__init__()
        if parsers is None:
            parsers = []
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
        if first_parse.parsed_intent is not None:
            return first_parse
        else:
            return self._parse(text, second_parsers)

    @staticmethod
    def _parse(text, parsers):
        if len(parsers) == 0:
            return Result(text, parsed_intent=None, parsed_entities=None)

        best_parser = None
        best_intent = None
        for parser in parsers:
            res = parser.get_intent(text)
            if best_intent is None or res.probability > best_intent.probability:
                best_intent = res
                best_parser = parser
        entities = best_parser.get_entities(text)
        return Result(text, parsed_intent=best_intent, parsed_entities=entities)

    def fit(self, dataset):
        validate_dataset(dataset)
        updated_parsers = []
        for intent_name in dataset["intents"].keys():
            parser = RegexIntentParser(intent_name).fit(dataset)
            updated_parsers.append(parser)
        self.custom_parsers = updated_parsers
        self.fitted = True
        return self

    def save_to_pickle_string(self):
        self.builtin_parsers = []
        return cPickle.dumps(self)

    @classmethod
    def load_from_dict(cls, obj_dict):
        return SnipsNLUEngine()

    @classmethod
    def load_from_pickle_and_path(cls, pkl_str, builtin_dir_path):
        engine = cPickle.loads(pkl_str)
        # TODO: update engine with builtin parsers using builtin_dir_path
        return engine

    @classmethod
    def load_from_pickle_and_byte_array(cls, pkl_str, builtin_byte_array):
        engine = cPickle.loads(pkl_str)
        # TODO: update engine with builtin parsers using builtin_byte_array
        return engine

    def __eq__(self, other):
        if self.fitted != other.fitted:
            return False
        for i, parser in enumerate(self.custom_parsers):
            if parser != other.custom_parsers[i]:
                return False
        return True
