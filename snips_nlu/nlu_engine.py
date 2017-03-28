import cPickle
from abc import ABCMeta, abstractmethod

from snips_nlu.built_in_entities import (BuiltInEntityLookupError,
                                         get_built_in_entity_by_label)
from snips_nlu.dataset import validate_dataset
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.builtin_intent_parser import BuiltinIntentParser
from snips_nlu.intent_parser.crf_intent_parser import CRFIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import Tagging
from snips_nlu.slot_filler.feature_functions import crf_features


class NLUEngine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        raise NotImplementedError


def _parse(text, parsers, keyword_entities_sets):
    if len(parsers) == 0:
        return Result(text, parsed_intent=None, parsed_slots=None)
    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue
        slots = parser.get_slots(text, res.intent_name)
        valid_slot = []
        for s in slots:
            if s.entity in keyword_entities_sets:
                if s.value not in keyword_entities_sets[s.entity]:
                    continue
            valid_slot.append(s)
        return Result(text, parsed_intent=res, parsed_slots=valid_slot)
    return Result(text, parsed_intent=None, parsed_slots=None)


def extract_keyword_entities_sets(dataset):
    keyword_entities = dict()
    for entity_name, entity in dataset["entities"].iteritems():
        if not entity["automatically_extensible"]:
            if entity["use_synonyms"]:
                keywords = set(s for d in entity["data"]
                               for s in d["synonyms"])
            else:
                keywords = set(d["data"]["value"] for d in entity["data"])

            keyword_entities[entity_name] = keywords
    return keyword_entities


def get_intent_custom_entities(dataset, intent):
    intent_entities = set()
    for utterance in dataset["intents"][intent]["utterances"]:
        for c in utterance["data"]:
            if "entity" in c:
                intent_entities.add(c["entity"])
    custom_entities = dict()
    for ent in intent_entities:
        try:
            get_built_in_entity_by_label(ent)
        except BuiltInEntityLookupError:
            custom_entities[ent] = dataset["entities"][ent]
    return custom_entities


class SnipsNLUEngine(NLUEngine):
    def __init__(self, custom_parsers=None, builtin_parser=None):
        super(SnipsNLUEngine, self).__init__()
        if custom_parsers is None:
            custom_parsers = []
        self.custom_parsers = custom_parsers
        self.builtin_parser = builtin_parser
        self.keyword_entities_sets = None

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        parsers = self.custom_parsers
        if self.builtin_parser is not None:
            parsers.append(self.builtin_parser)
        return _parse(text, parsers, self.keyword_entities_sets)

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
        self.keyword_entities_sets = extract_keyword_entities_sets(dataset)
        taggers = dict()
        for intent in dataset["intents"].keys():
            intent_entities = get_intent_custom_entities(dataset, intent)
            features = crf_features(intent_entities,
                                    language=dataset["language"])
            taggers[intent] = CRFTagger(default_crf_model(), features,
                                        Tagging.BILOU)
        crf_parser = CRFIntentParser(intent_classifier, taggers).fit(dataset)
        self.custom_parsers = [custom_parser, crf_parser]
        return self

    def to_dict(self):
        """
        Serialize the SnipsNLUEngine to a json dict, after having reset the
        builtin intent parser. Thus this serialization, contains only the
        custom intent parsers.
        """
        return {
            "custom_parsers": [p.to_dict() for p in self.custom_parsers],
            "builtin_parser": None
        }

    @staticmethod
    def from_dict(obj_dict):
        custom_parsers = [IntentParser.from_dict(d) for d in
                          obj_dict["custom_parsers"]]
        builtin_parser = None
        if "builtin_parser" in obj_dict \
                and obj_dict["builtin_parser"] is not None:
            builtin_parser = BuiltinIntentParser.from_dict(
                obj_dict["builtin_parser"])
        return SnipsNLUEngine(custom_parsers=custom_parsers,
                              builtin_parser=builtin_parser)
