from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from copy import deepcopy

from snips_nlu.builtin_entities import BuiltInEntity, is_builtin_entity
from snips_nlu.config import SlotFillerDataAugmentationConfig, NLUConfig
from snips_nlu.constants import (
    INTENTS, ENTITIES, UTTERANCES, LANGUAGE, AUTOMATICALLY_EXTENSIBLE,
    ENTITY, DATA, SLOT_NAME, CAPITALIZE)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser, fit_tagger
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot, empty_result, \
    IntentClassificationResult
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_tagger import CRFTagger, get_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_functions import crf_features
from snips_nlu.utils import check_random_state


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
    def parse(self, text, intent=None):
        pass


def _parse(text, entities, rule_based_parser=None, probabilistic_parser=None,
           intent=None):
    parsers = []
    if rule_based_parser is not None:
        parsers.append(rule_based_parser)
    if probabilistic_parser is not None:
        parsers.append(probabilistic_parser)

    if not parsers:
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
        mapping = dict()
        slot_name_mapping[intent_name] = mapping
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


def get_intent_slot_name_mapping(dataset, intent):
    slot_name_mapping = dict()
    intent_data = dataset[INTENTS][intent]
    for utterance in intent_data[UTTERANCES]:
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name_mapping[chunk[SLOT_NAME]] = chunk[ENTITY]
    return slot_name_mapping


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


def is_trainable_regex_intent(intent, entities, regex_training_config):
    if len(intent[UTTERANCES]) >= regex_training_config.max_queries:
        return False

    intent_entities = set(chunk[ENTITY] for query in intent[UTTERANCES]
                          for chunk in query[DATA] if ENTITY in chunk)
    intent_entities = [ent for ent in intent_entities
                       if not is_builtin_entity(ent)]
    total_entities = sum(len(entities[entity_name][UTTERANCES])
                         for entity_name in intent_entities)
    if total_entities > regex_training_config.max_entities:
        return False
    return True


class SnipsNLUEngine(NLUEngine):
    def __init__(self, language, config=None, rule_based_parser=None,
                 probabilistic_parser=None, entities=None,
                 slot_name_mapping=None, intents_data_sizes=None,
                 random_seed=None):
        super(SnipsNLUEngine, self).__init__(language)
        self._config = None
        if config is None:
            config = NLUConfig()
        self.config = config

        self.rule_based_parser = rule_based_parser
        self.probabilistic_parser = probabilistic_parser
        if entities is None:
            entities = dict()

        self.entities = entities
        if slot_name_mapping is None:
            slot_name_mapping = dict()

        self.slot_name_mapping = slot_name_mapping
        self.intents_data_sizes = intents_data_sizes
        self._pre_trained_taggers = dict()
        self.random_seed = random_seed

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, NLUConfig):
            config = value
        elif isinstance(value, dict):
            config = NLUConfig.from_dict(value)
        else:
            raise TypeError("Expected config to be a dict or a NLUConfig")
        self._config = config

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
        - intents: list of intents to train. If `None`, all intents will be
        trained. This parameter allows to have pre-trained intents.

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

        if missing_intents:
            raise ValueError(
                "These intents must be trained: %s" % missing_intents)

        dataset = validate_and_format_dataset(dataset)
        self.entities = dict()
        for entity_name, entity in dataset[ENTITIES].iteritems():
            if is_builtin_entity(entity_name):
                continue
            ent = deepcopy(entity)
            ent.pop(CAPITALIZE)
            self.entities[entity_name] = ent

        regex_intents = [
            intent_name for intent_name, intent in dataset[INTENTS].iteritems()
            if is_trainable_regex_intent(
                intent, self.entities, self.config.regex_training_config)]

        self.rule_based_parser = RegexIntentParser(self.language).fit(
            dataset, intents=regex_intents)

        self.intents_data_sizes = {intent_name: len(intent[UTTERANCES])
                                   for intent_name, intent
                                   in dataset[INTENTS].iteritems()}
        self.slot_name_mapping = get_slot_name_mapping(dataset)

        random_state = check_random_state(self.random_seed)
        taggers = dict()
        for intent in dataset[INTENTS]:
            features_config = self.config.probabilistic_intent_parser_config \
                .crf_features_config
            features = crf_features(dataset, intent, self.language,
                                    features_config, random_state)
            if intent in self._pre_trained_taggers:
                tagger = self._pre_trained_taggers[intent]
            else:
                tagger = CRFTagger(get_crf_model(), features,
                                   TaggingScheme.BIO, self.language)
            taggers[intent] = tagger
        intent_classifier = SnipsIntentClassifier(
            self.language, self.config.intent_classifier_config,
            random_seed=self.random_seed)
        self.probabilistic_parser = ProbabilisticIntentParser(
            self.language,
            intent_classifier,
            taggers,
            self.slot_name_mapping,
            self.config.probabilistic_intent_parser_config,
            random_seed=self.random_seed
        )
        self.probabilistic_parser.fit(dataset, intents=intents)
        self._pre_trained_taggers = taggers
        return self

    def get_fitted_tagger(self, dataset, intent):
        dataset = validate_and_format_dataset(dataset)
        crf_features_config = self.config.probabilistic_intent_parser_config \
            .crf_features_config
        random_state = check_random_state(self.random_seed)
        features = crf_features(dataset, intent, self.language,
                                crf_features_config, random_state)
        tagger = CRFTagger(get_crf_model(), features, TaggingScheme.BIO,
                           self.language)
        if self.probabilistic_parser is not None:
            config = self.probabilistic_parser.data_augmentation_config
        else:
            config = SlotFillerDataAugmentationConfig()
        return fit_tagger(tagger, dataset, intent, self.language, config,
                          random_state)

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
            "config": self.config.to_dict(),
            "random_seed": self.random_seed
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

        rule_based_parser = None
        probabilistic_parser = None

        if "rule_based_parser" in obj_dict["model"]:
            rule_based_parser = RegexIntentParser.from_dict(
                obj_dict["model"]["rule_based_parser"])

        if "probabilistic_parser" in obj_dict["model"]:
            probabilistic_parser = ProbabilisticIntentParser.from_dict(
                obj_dict["model"]["probabilistic_parser"])

        return cls(
            language=language, rule_based_parser=rule_based_parser,
            probabilistic_parser=probabilistic_parser, entities=entities,
            slot_name_mapping=slot_name_mapping,
            intents_data_sizes=intents_data_sizes,
            config=obj_dict["config"],
            random_seed=obj_dict["random_seed"]
        )
