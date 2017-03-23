import cPickle
from abc import ABCMeta, abstractmethod

from snips_nlu.dataset import validate_dataset
from snips_nlu.intent_parser.builtin_intent_parser import BuiltinIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.result import Result


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        raise NotImplementedError


def _parse(text, parsers):
    if len(parsers) == 0:
        return Result(text, parsed_intent=None, parsed_entities=None)
    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue
        entities = parser.get_entities(text, res.intent_name)
        return Result(text, parsed_intent=res, parsed_entities=entities)
    return Result(text, parsed_intent=None, parsed_entities=None)


class SnipsNLUEngine(NLUEngine):
    def __init__(self, custom_parsers=None, builtin_parser=None):
        super(SnipsNLUEngine, self).__init__()
        if custom_parsers is None:
            custom_parsers = []
        self.custom_parsers = custom_parsers
        self.builtin_parser = builtin_parser

    def parse(self, text):
        parsers = self.custom_parsers
        if self.builtin_parser is not None:
            parsers.append(self.builtin_parser)
        return _parse(text, parsers)

    def fit(self, dataset):
        validate_dataset(dataset)
        custom_parser = RegexIntentParser().fit(dataset)
        self.custom_parsers = [custom_parser]
        return self

    def save_to_pickle_string(self):
        self.builtin_parser = None
        return cPickle.dumps(self)

    @classmethod
    def load_from_pickle_and_path(cls, pkl_str, builtin_path):
        engine = cPickle.loads(pkl_str)
        engine.builtin_parser = BuiltinIntentParser(builtin_path)
        return engine

    @classmethod
    def load_from_pickle_and_byte_array(cls, pkl_str, builtin_byte_array):
        engine = cPickle.loads(pkl_str)
        # TODO: update engine with builtin parsers using builtin_byte_array
        return engine
