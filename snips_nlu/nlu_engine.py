from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod

from duckling import core

from dataset import validate_and_format_dataset
from snips_nlu.built_in_entities import BuiltInEntity, get_builtin_entities, \
    is_builtin_entity
from snips_nlu.constants import (
    INTENTS, ENTITIES, UTTERANCES, LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE,
    ENTITY, MATCH_RANGE, DATA, SLOT_NAME,
    USE_SYNONYMS, SYNONYMS, TOKEN_INDEXES, NGRAM)
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot, empty_result, \
    IntentClassificationResult
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_functions import crf_features
from snips_nlu.slot_filler.features_utils import get_all_ngrams
from snips_nlu.tokenization import tokenize


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


def _tag_seen_entities(text, entities):
    # TODO handle case properly but can be tricky with the synonyms mapping
    tokens = tokenize(text)
    ngrams = get_all_ngrams([t.value for t in tokens])
    ngrams = sorted(ngrams, key=lambda ng: len(ng[TOKEN_INDEXES]),
                    reverse=True)

    slots = []
    for ngram in ngrams:
        ngram_slots = []
        str_ngram = ngram[NGRAM]
        matched_several_entities = False
        for entity_name, entity_data in entities.iteritems():
            if str_ngram in entity_data[UTTERANCES]:
                if len(ngram_slots) == 1:
                    matched_several_entities = True
                    break
                rng = (tokens[min(ngram[TOKEN_INDEXES])].start,
                       tokens[max(ngram[TOKEN_INDEXES])].end)
                value = entity_data[UTTERANCES][str_ngram]
                ngram_slots.append(
                    ParsedSlot(rng, value, entity_name, entity_name))
        if not matched_several_entities:
            slots = enrich_slots(slots, ngram_slots)

    return slots


def _parse(text, entities, rule_based_parser=None, probabilistic_parser=None,
           intent=None):
    parsers = []
    if rule_based_parser is not None:
        parsers.append(rule_based_parser)
    if probabilistic_parser is not None:
        parsers.append(probabilistic_parser)

    if len(parsers) == 0:
        return empty_result(text)

    for parser in parsers:
        if intent is None:
            res = parser.get_intent(text)
            if res is None:
                continue
            intent_name = res.intent_name
        else:
            res = IntentClassificationResult(intent, 1.0)
            intent_name = intent
        valid_slot = []
        slots = parser.get_slots(text, intent_name)
        for s in slots:
            slot_value = s.value
            # Check if the entity is from a custom intent
            if s.entity in entities:
                entity = entities[s.entity]
                if not entity[AUTOMATICALLY_EXTENSIBLE]:
                    if s.value not in entity[UTTERANCES]:
                        continue
                    slot_value = entity[UTTERANCES][s.value]
            s = ParsedSlot(s.match_range, slot_value, s.entity,
                           s.slot_name)
            valid_slot.append(s)
        return Result(text, parsed_intent=res, parsed_slots=valid_slot)
    return empty_result(text)


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


class SnipsNLUEngine(NLUEngine):
    def __init__(self, language, rule_based_parser=None,
                 probabilistic_parser=None, entities=None,
                 slot_name_mapping=None, tagging_threshold=None,
                 intents_data_sizes=None):
        super(SnipsNLUEngine, self).__init__(language)
        self.rule_based_parser = rule_based_parser
        self.probabilistic_parser = probabilistic_parser
        if entities is None:
            entities = dict()
        self.entities = entities
        if slot_name_mapping is None:
            slot_name_mapping = dict()
        self.slot_name_mapping = slot_name_mapping
        if tagging_threshold is None:
            tagging_threshold = 5
        self.tagging_threshold = tagging_threshold
        self.intents_data_sizes = intents_data_sizes
        self._pre_trained_taggers = dict()
        self.tagging_scope = []
        tagging_excluded_entities = {BuiltInEntity.NUMBER}
        for d in core.get_dims(self.language.iso_code):
            ent = BuiltInEntity.built_in_entity_by_duckling_dim.get(d, False)
            if ent and ent not in tagging_excluded_entities:
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

    def tag(self, text, intent):
        """
        Parse the input text conditionally to the knowledge of `intent`.
        This method is more aggressive (less conservative) than `parse`.
        """
        result = self._parse(text, intent=intent)
        enrich_results = self.intents_data_sizes[
                             intent] < self.tagging_threshold
        if not enrich_results:
            return result

        # Add slots seen in other queries from other intents
        seen_entities_slots = _tag_seen_entities(text, self.entities)

        # Add builtins entities
        builtin_entities = get_builtin_entities(text, self.language,
                                                self.tagging_scope)
        builtin_slots = [ParsedSlot(ent[MATCH_RANGE], ent[VALUE],
                                    ent[ENTITY].label, ent[ENTITY].label)
                         for ent in builtin_entities]
        slots = enrich_slots(seen_entities_slots, builtin_slots)

        # Add current model results
        slots = enrich_slots(slots, result.parsed_slots)

        slots = sorted(slots, key=lambda x: x.match_range[0])

        parsed_intent = IntentClassificationResult(
            result.parsed_intent.intent_name, result.parsed_intent.probability)
        return Result(text, parsed_intent=parsed_intent,
                      parsed_slots=slots).as_dict()

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
        self.rule_based_parser = RegexIntentParser(self.language).fit(dataset)
        self.entities = snips_nlu_entities(dataset)
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

    def add_pretrained_model(self, intent, model_data):
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
            "tagging_threshold": self.tagging_threshold,
            ENTITIES: self.entities,
            "intents_data_sizes": self.intents_data_sizes,
            "model": model_dict
        }

    @classmethod
    def from_dict(cls, obj_dict):
        """
        Loads a SnipsNLUEngine instance from a python dictionary.
        """
        language = Language.from_iso_code(obj_dict[LANGUAGE])
        slot_name_mapping = obj_dict["slot_name_mapping"]
        tagging_threshold = obj_dict["tagging_threshold"]
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

        return SnipsNLUEngine(
            language=language, rule_based_parser=rule_based_parser,
            probabilistic_parser=probabilistic_parser, entities=entities,
            slot_name_mapping=slot_name_mapping,
            tagging_threshold=tagging_threshold,
            intents_data_sizes=intents_data_sizes
        )
