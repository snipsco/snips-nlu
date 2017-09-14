from __future__ import unicode_literals

import argparse
import io
import json
from abc import ABCMeta, abstractmethod

from dataset import validate_and_format_dataset
from snips_nlu.builtin_entities import BuiltInEntity, is_builtin_entity, \
    _SUPPORTED_BUILTINS_BY_LANGUAGE
from snips_nlu.constants import (
    INTENTS, ENTITIES, UTTERANCES, LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE,
    ENTITY, DATA, SLOT_NAME,
    USE_SYNONYMS, SYNONYMS)
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser, fit_tagger
from snips_nlu.data_augmentation import DataAugmentationConfig
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot, empty_result, \
    IntentClassificationResult
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_functions import crf_features


class NLUEngine(object):
    __metaclass__ = ABCMeta

    def __init__(self, language):
        self._language = None
        self.language = language

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if isinstance(value, Language):
            self._language = value
        elif isinstance(value, (str, unicode)):
            self._language = Language.from_iso_code(value)
        else:
            raise TypeError("Expected str, unicode or Language found '%s'"
                            % type(value))

    @abstractmethod
    def parse(self, text):
        pass


def _parse(text, entities, rule_based_parser=None, probabilistic_parser=None,
           intent=None):
    parsers = []
    if rule_based_parser is not None:
        parsers.append(rule_based_parser)
    if probabilistic_parser is not None:
        parsers.append(probabilistic_parser)

    if len(parsers) == 0:
        return empty_result(text)

    result = empty_result(text) if intent is None else Result(
        text, parsed_intent=IntentClassificationResult(intent, 1.0),
        parsed_slots=[])

    for parser in parsers:
        res = parser.get_intent(text)
        if res is None:
            continue

        intent_name = res.intent_name
        if intent is not None:
            if intent_name != intent:
                continue
            res = IntentClassificationResult(intent_name, 1.0)

        valid_slot = []
        slots = parser.get_slots(text, intent_name)
        for s in slots:
            slot_value = s.value
            # Check if the entity is from a custom intent
            if s.entity in entities:
                entity = entities[s.entity]
                if s.value in entity[UTTERANCES]:
                    slot_value = entity[UTTERANCES][s.value]
                elif not entity[AUTOMATICALLY_EXTENSIBLE]:
                    continue
            s = ParsedSlot(s.match_range, slot_value, s.entity,
                           s.slot_name)
            valid_slot.append(s)
        return Result(text, parsed_intent=res, parsed_slots=valid_slot)
    return result


def spans_to_tokens_indexes(spans, tokens):
    tokens_indexes = []
    for span_start, span_end in spans:
        indexes = []
        for i, token in enumerate(tokens):
            if span_end > token.start and span_start < token.end:
                indexes.append(i)
        tokens_indexes.append(indexes)
    return tokens_indexes


def get_slot_name_mapping(dataset):
    """
    Returns a dict which maps slot names to entities
    """
    slot_name_mapping = dict()
    for intent_name, intent in dataset[INTENTS].iteritems():
        _dict = dict()
        slot_name_mapping[intent_name] = _dict
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    _dict[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def get_intent_slot_name_mapping(dataset, intent):
    slot_name_mapping = dict()
    intent_data = dataset[INTENTS][intent]
    for utterance in intent_data[UTTERANCES]:
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name_mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def get_intent_custom_entities(dataset, intent):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent][UTTERANCES]:
        for c in utterance[DATA]:
            if ENTITY in c:
                intent_entities.add(c[ENTITY])
    custom_entities = dict()
    for ent in intent_entities:
        if ent not in BuiltInEntity.built_in_entity_by_label:
            custom_entities[ent] = dataset[ENTITIES][ent]
    return custom_entities


def snips_nlu_entities(dataset):
    entities = dict()
    for entity_name, entity in dataset[ENTITIES].iteritems():
        if is_builtin_entity(entity_name):
            continue
        entity_data = dict()
        use_synonyms = entity[USE_SYNONYMS]
        automatically_extensible = entity[AUTOMATICALLY_EXTENSIBLE]
        entity_data[AUTOMATICALLY_EXTENSIBLE] = automatically_extensible

        entity_utterances = dict()
        for data in entity[DATA]:
            if use_synonyms:
                for s in data[SYNONYMS]:
                    entity_utterances[s] = data[VALUE]
            else:
                entity_utterances[data[VALUE]] = data[VALUE]
        entity_data[UTTERANCES] = entity_utterances
        entities[entity_name] = entity_data
    return entities


def enrich_slots(slots, other_slots):
    enriched_slots = list(slots)
    for slot in other_slots:
        if any((slot.match_range[1] > s.match_range[0])
               and (slot.match_range[0] < s.match_range[1])
               for s in enriched_slots):
            continue
        enriched_slots.append(slot)
    return enriched_slots


TAGGING_EXCLUDED_ENTITIES = {BuiltInEntity.NUMBER}


def is_trainable_regex_intent(intent, entities, num_queries_threshold,
                              num_entities_threshold):
    if len(intent[UTTERANCES]) >= num_queries_threshold:
        return False

    intent_entities = set(chunk[ENTITY] for query in intent[UTTERANCES]
                          for chunk in query[DATA] if ENTITY in chunk)
    intent_entities = [ent for ent in intent_entities
                       if not is_builtin_entity(ent)]
    if sum(len(entities[entity_name][UTTERANCES])
           for entity_name in intent_entities) > num_entities_threshold:
        return False
    return True


class SnipsNLUEngine(NLUEngine):
    def __init__(self, language, rule_based_parser=None,
                 probabilistic_parser=None, entities=None,
                 slot_name_mapping=None, intents_data_sizes=None,
                 num_queries_threshold=50, num_entities_threshold=200):
        super(SnipsNLUEngine, self).__init__(language)
        self.rule_based_parser = rule_based_parser
        self.probabilistic_parser = probabilistic_parser
        if entities is None:
            entities = dict()
        self.entities = entities
        if slot_name_mapping is None:
            slot_name_mapping = dict()
        self.slot_name_mapping = slot_name_mapping
        self.intents_data_sizes = intents_data_sizes
        self.num_queries_threshold = num_queries_threshold
        self.num_entities_threshold = num_entities_threshold
        self._pre_trained_taggers = dict()
        self.tagging_scope = []
        for ent in _SUPPORTED_BUILTINS_BY_LANGUAGE[self.language]:
            if ent and ent not in TAGGING_EXCLUDED_ENTITIES:
                self.tagging_scope.append(ent)

    def parse(self, text, intent=None):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        return self._parse(text, intent=intent).as_dict()

    def _parse(self, text, intent=None):
        return _parse(text, self.entities, self.rule_based_parser,
                      self.probabilistic_parser, intent)

    def fit(self, dataset, intents=None):

        """
        Fit the NLU engine.

        Parameters
        ----------
        - dataset: dict containing intents and entities data
        - intents: list of intents to train. If `None`, all intents will 
        be trained. This parameter allows to have pre-trained intents.

        Returns
        -------
        The same object, trained
        """
        all_intents = set(dataset[INTENTS].keys())
        if intents is None:
            intents = all_intents
        else:
            intents = set(intents)

        implicit_pretrained_intents = all_intents.difference(intents)
        actual_pretrained_intents = set(self._pre_trained_taggers.keys())
        missing_intents = implicit_pretrained_intents.difference(
            actual_pretrained_intents)

        if len(missing_intents) > 0:
            raise ValueError(
                "These intents must be trained: %s" % missing_intents)

        dataset = validate_and_format_dataset(dataset)

        self.entities = snips_nlu_entities(dataset)

        regex_intents = [
            intent_name for intent_name, intent in dataset[INTENTS].iteritems()
            if is_trainable_regex_intent(intent, self.entities,
                                         self.num_queries_threshold,
                                         self.num_entities_threshold)]

        self.rule_based_parser = RegexIntentParser(self.language).fit(
            dataset, intents=regex_intents)

        self.intents_data_sizes = {intent_name: len(intent[UTTERANCES])
                                   for intent_name, intent
                                   in dataset[INTENTS].iteritems()}
        self.slot_name_mapping = get_slot_name_mapping(dataset)
        taggers = dict()
        for intent in dataset[INTENTS]:
            intent_custom_entities = get_intent_custom_entities(dataset,
                                                                intent)
            features = crf_features(intent_custom_entities, self.language)
            if intent in self._pre_trained_taggers:
                tagger = self._pre_trained_taggers[intent]
            else:
                tagger = CRFTagger(default_crf_model(), features,
                                   TaggingScheme.BIO, self.language)
            taggers[intent] = tagger
        intent_classifier = SnipsIntentClassifier(self.language)
        self.probabilistic_parser = ProbabilisticIntentParser(
            self.language, intent_classifier, taggers, self.slot_name_mapping)
        self.probabilistic_parser.fit(dataset, intents=intents)
        self._pre_trained_taggers = taggers
        return self

    def get_fitted_tagger(self, dataset, intent):
        dataset = validate_and_format_dataset(dataset)
        intent_custom_entities = get_intent_custom_entities(dataset, intent)
        features = crf_features(intent_custom_entities, self.language)
        tagger = CRFTagger(default_crf_model(), features, TaggingScheme.BIO,
                           self.language)
        if self.probabilistic_parser is not None:
            config = self.probabilistic_parser.data_augmentation_config
        else:
            config = DataAugmentationConfig()
        return fit_tagger(tagger, dataset, intent, self.language, config)

    def add_fitted_tagger(self, intent, model_data):
        tagger = CRFTagger.from_dict(model_data)
        if self.probabilistic_parser is not None:
            self.probabilistic_parser.crf_taggers[intent] = tagger
        self._pre_trained_taggers[intent] = tagger

    def to_dict(self):
        """
        Serialize the nlu engine into a python dictionary
        """
        model_dict = dict()
        if self.rule_based_parser is not None:
            model_dict["rule_based_parser"] = self.rule_based_parser.to_dict()
        if self.probabilistic_parser is not None:
            model_dict["probabilistic_parser"] = \
                self.probabilistic_parser.to_dict()

        return {
            LANGUAGE: self.language.iso_code,
            "slot_name_mapping": self.slot_name_mapping,
            ENTITIES: self.entities,
            "intents_data_sizes": self.intents_data_sizes,
            "model": model_dict,
            "num_queries_threshold": self.num_queries_threshold,
            "num_entities_threshold": self.num_entities_threshold
        }

    @classmethod
    def from_dict(cls, obj_dict):
        """
        Loads a SnipsNLUEngine instance from a python dictionary.
        """
        language = Language.from_iso_code(obj_dict[LANGUAGE])
        slot_name_mapping = obj_dict["slot_name_mapping"]
        entities = obj_dict[ENTITIES]
        intents_data_sizes = obj_dict["intents_data_sizes"]
        num_queries_threshold = obj_dict["num_queries_threshold"]
        num_entities_threshold = obj_dict["num_entities_threshold"]

        rule_based_parser = None
        probabilistic_parser = None

        if "rule_based_parser" in obj_dict["model"]:
            rule_based_parser = RegexIntentParser.from_dict(
                obj_dict["model"]["rule_based_parser"])

        if "probabilistic_parser" in obj_dict["model"]:
            probabilistic_parser = ProbabilisticIntentParser.from_dict(
                obj_dict["model"]["probabilistic_parser"])

        return SnipsNLUEngine(
            language=language, rule_based_parser=rule_based_parser,
            probabilistic_parser=probabilistic_parser, entities=entities,
            slot_name_mapping=slot_name_mapping,
            intents_data_sizes=intents_data_sizes,
            num_queries_threshold=num_queries_threshold,
            num_entities_threshold=num_entities_threshold
        )


def main_create_and_train_engine():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=unicode)
    parser.add_argument("dataset_path", type=unicode)
    parser.add_argument("output_path", type=unicode)
    args = vars(parser.parse_args())

    dataset_path = args.pop("dataset_path")
    with io.open(dataset_path, "r", encoding="utf8") as f:
        dataset = json.load(f)

    language = Language.from_iso_code(args.pop("language"))
    engine = SnipsNLUEngine(language).fit(dataset)
    print("Create and trained the engine...")

    output_path = args.pop("output_path")
    serialized_engine = json.dumps(engine.to_dict()).decode("utf8")
    with io.open(output_path, "w", encoding="utf8") as f:
        f.write(serialized_engine)
    print("Saved the trained engine to %s" % output_path)
