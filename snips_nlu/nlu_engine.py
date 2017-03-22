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


def _parse(text, parsers, threshold):
    if len(parsers) == 0:
        return Result(text, parsed_intent=None, parsed_entities=None)

    best_parser = None
    best_intent = None
    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue
        if best_intent is None or res.probability > best_intent.probability:
            best_intent = res
            best_parser = parser
    if best_intent is None or best_intent.probability <= threshold:
        return Result(text, parsed_intent=None, parsed_entities=None)
    entities = best_parser.get_entities(text, best_intent.intent_name)
    return Result(text, parsed_intent=best_intent, parsed_entities=entities)


class SnipsNLUEngine(NLUEngine):
    def __init__(self, custom_parsers=None, builtin_parser=None):
        super(SnipsNLUEngine, self).__init__()
        if custom_parsers is None:
            custom_parsers = []
        self.custom_parsers = custom_parsers
        self.builtin_parser = builtin_parser
        self.fitted = False

    def parse(self, text):
        custom_parse = _parse(text, self.custom_parsers, threshold=0.)
        if custom_parse.parsed_intent is not None:
            return custom_parse
        elif self.builtin_parser is not None:
            return _parse(text, [self.builtin_parser], threshold=0.)
        else:
            return Result(text=text, parsed_intent=None, parsed_entities=None)

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
        self.builtin_parser = None
        return cPickle.dumps(self)

    @classmethod
    def load_from_dict(cls, obj_dict):
        return SnipsNLUEngine()

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
