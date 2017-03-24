import cPickle
from abc import ABCMeta, abstractmethod

from snips_nlu.dataset import validate_dataset
from snips_nlu.intent_classifier.intent_classifier import SnipsIntentClassifier
from snips_nlu.intent_parser.builtin_intent_parser import BuiltinIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.intent_parser.crf_intent_parser import CRFIntentParser
from snips_nlu.slot_filler.crf_slot_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import Tagging
from snips_nlu.slot_filler.feature_functions import default_features
from snips_nlu.result import Result


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        raise NotImplementedError


def _parse(text, parsers):
    if len(parsers) == 0:
        return Result(text, parsed_intent=None, parsed_slots=None)
    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue
        slots = parser.get_slots(text, res.intent_name)
        return Result(text, parsed_intent=res, parsed_slots=slots)
    return Result(text, parsed_intent=None, parsed_slots=None)


class SnipsNLUEngine(NLUEngine):
    def __init__(self, custom_parsers=None, builtin_parser=None):
        super(SnipsNLUEngine, self).__init__()
        if custom_parsers is None:
            custom_parsers = []
        self.custom_parsers = custom_parsers
        self.builtin_parser = builtin_parser

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        parsers = self.custom_parsers
        if self.builtin_parser is not None:
            parsers.append(self.builtin_parser)
        return _parse(text, parsers)

    def fit(self, dataset):
        """
        Fit the engine with a dataset
        :param dataset: A dictionary containing the data of the custom intents.
        See https://github.com/snipsco/snips-nlu/blob/develop/README.md for
        details about the format.
        :return: A fitted SnipsNLUEngine
        """
        validate_dataset(dataset)
        custom_parser = RegexIntentParser().fit(dataset)
        intent_classifier = SnipsIntentClassifier().fit(dataset)
        taggers = {}
        for intent in dataset["intents"].keys():
            taggers[intent] = CRFTagger(default_crf_model(), default_features(),
                                        Tagging.BILOU)
        crf_parser = CRFIntentParser(intent_classifier, taggers).fit(dataset)
        self.custom_parsers = [custom_parser, crf_parser]
        return self

    def save_to_pickle_string(self):
        """
        Serialize the SnipsNLUEngine to a pickle string, after having reset the
        builtin intent parser. Thus this serialization, contains only the
        custom intent parsers.
        """
        self.builtin_parser = None
        return cPickle.dumps(self)

    @classmethod
    def load_from_pickle_and_path(cls, pkl_str, builtin_path):
        """
        :param pkl_str: content of the pickle file describing the nlu engine
        :param builtin_path: path of the builtin intent parser directory
        :return: a SnipsNLUEngine already fitted
        """
        engine = cPickle.loads(pkl_str)
        engine.builtin_parser = BuiltinIntentParser(builtin_path)
        return engine

    @classmethod
    def load_from_pickle_and_byte_array(cls, pkl_str, builtin_byte_array):
        """
        :param pkl_str: content of the pickle file describing the nlu engine
        :param builtin_byte_array: byte_array data need to initialize the
        builtin intent parser
        :return:
        """
        engine = cPickle.loads(pkl_str)
        # TODO: update engine with builtin parsers using builtin_byte_array
        return engine
