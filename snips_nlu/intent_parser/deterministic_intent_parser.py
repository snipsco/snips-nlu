from __future__ import unicode_literals

import re
from copy import deepcopy

from future.utils import itervalues, iteritems

from snips_nlu.builtin_entities import is_builtin_entity, \
    get_builtin_entities
from snips_nlu.constants import (
    TEXT, DATA, INTENTS, ENTITIES, SLOT_NAME, UTTERANCES, ENTITY,
    RES_MATCH_RANGE, LANGUAGE, RES_INTENT, RES_SLOTS, RES_VALUE)
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.languages import Language
from snips_nlu.pipeline.configs.intent_parser import \
    DeterministicIntentParserConfig
from snips_nlu.result import (_slot, parsing_result,
                              intent_classification_result, empty_result)
from snips_nlu.tokenization import tokenize, tokenize_light
from snips_nlu.utils import regex_escape, ranges_overlap

GROUP_NAME_PREFIX = "group"
GROUP_NAME_SEPARATOR = "_"


def get_index(index):
    split = index.split(GROUP_NAME_SEPARATOR)
    if len(split) != 2 or split[0] != GROUP_NAME_PREFIX:
        raise ValueError("Misformatted index")
    return int(split[1])


def make_index(i):
    return "%s%s%s" % (GROUP_NAME_PREFIX, GROUP_NAME_SEPARATOR, i)


def generate_new_index(slots_name_to_labels):
    if not slots_name_to_labels:
        index = make_index(0)
    else:
        max_index = max(slots_name_to_labels, key=get_index)
        max_index = get_index(max_index) + 1
        index = make_index(max_index)
    return index


def get_slot_names_mapping(dataset):
    slot_names_to_entities = dict()
    for intent in itervalues(dataset[INTENTS]):
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    slot_name = chunk[SLOT_NAME]
                    entity = chunk[ENTITY]
                    slot_names_to_entities[slot_name] = entity
    return slot_names_to_entities


def query_to_pattern(query, joined_entity_utterances,
                     group_names_to_slot_names, language):
    pattern = []
    for chunk in query[DATA]:
        if SLOT_NAME in chunk:
            max_index = generate_new_index(group_names_to_slot_names)
            slot_name = chunk[SLOT_NAME]
            entity = chunk[ENTITY]
            group_names_to_slot_names[max_index] = slot_name
            pattern.append(
                r"(?P<%s>%s)" % (max_index, joined_entity_utterances[entity]))
        else:
            tokens = tokenize_light(chunk[TEXT], language)
            pattern += [regex_escape(t) for t in tokens]
    pattern = r"^%s%s%s$" % (language.ignored_characters_pattern,
                             language.ignored_characters_pattern.join(pattern),
                             language.ignored_characters_pattern)
    return pattern, group_names_to_slot_names


def get_queries_with_unique_context(intent_queries, language):
    contexts = set()
    queries = []
    for query in intent_queries:
        context = ""
        for chunk in query[DATA]:
            if ENTITY not in chunk:
                context += chunk[TEXT]
            else:
                context += get_builtin_entity_name(chunk[ENTITY], language)
        if context not in contexts:
            queries.append(query)
    return queries


def generate_regexes(intent_queries, joined_entity_utterances,
                     group_names_to_labels, language):
    queries = get_queries_with_unique_context(intent_queries, language)
    # Join all the entities utterances with a "|" to create the patterns
    patterns = set()
    for query in queries:
        pattern, group_names_to_labels = query_to_pattern(
            query, joined_entity_utterances, group_names_to_labels, language)
        patterns.add(pattern)
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    return regexes, group_names_to_labels


def get_joined_entity_utterances(dataset, language):
    joined_entity_utterances = dict()
    for entity_name, entity in iteritems(dataset[ENTITIES]):
        if is_builtin_entity(entity_name):
            utterances = [get_builtin_entity_name(entity_name, language)]
        else:
            utterances = list(entity[UTTERANCES])
        utterances_patterns = map(regex_escape, utterances)
        utterances_patterns = (p for p in utterances_patterns if p)
        joined_entity_utterances[entity_name] = r"|".join(
            sorted(utterances_patterns, key=len, reverse=True))
    return joined_entity_utterances


def deduplicate_overlapping_slots(slots, language):
    deduplicated_slots = []
    for slot in slots:
        is_overlapping = False
        for slot_index, dedup_slot in enumerate(deduplicated_slots):
            if ranges_overlap(slot[RES_MATCH_RANGE],
                              dedup_slot[RES_MATCH_RANGE]):
                is_overlapping = True
                tokens = tokenize(slot[RES_VALUE], language)
                dedup_tokens = tokenize(dedup_slot[RES_VALUE], language)
                if len(tokens) > len(dedup_tokens):
                    deduplicated_slots[slot_index] = slot
                elif len(tokens) == len(dedup_tokens) \
                        and len(slot[RES_VALUE]) > len(dedup_slot[RES_VALUE]):
                    deduplicated_slots[slot_index] = slot
        if not is_overlapping:
            deduplicated_slots.append(slot)
    return deduplicated_slots


def get_builtin_entity_name(entity_label, language):
    return "%%%s%%" % "".join(
        tokenize_light(entity_label, language)).upper()


def preprocess_builtin_entities(utterance, language):
    new_utterance = deepcopy(utterance)
    for i, chunk in enumerate(utterance[DATA]):
        _, processed_chunk_text = replace_builtin_entities(chunk[TEXT],
                                                           language)
        new_utterance[DATA][i][TEXT] = processed_chunk_text
    return new_utterance


def replace_builtin_entities(text, language):
    builtin_entities = get_builtin_entities(text, language)
    if not builtin_entities:
        return dict(), text

    range_mapping = dict()
    processed_text = ""
    offset = 0
    current_ix = 0
    builtin_entities = sorted(builtin_entities,
                              key=lambda e: e[RES_MATCH_RANGE][0])
    for ent in builtin_entities:
        ent_start = ent[RES_MATCH_RANGE][0]
        ent_end = ent[RES_MATCH_RANGE][1]
        rng_start = ent_start + offset

        processed_text += text[current_ix:ent_start]

        entity_length = ent[RES_MATCH_RANGE][1] - ent[RES_MATCH_RANGE][0]
        entity_place_holder = get_builtin_entity_name(ent[ENTITY].label,
                                                      language)

        offset += len(entity_place_holder) - entity_length

        processed_text += entity_place_holder
        rng_end = ent_end + offset
        new_range = (rng_start, rng_end)
        range_mapping[new_range] = ent[RES_MATCH_RANGE]
        current_ix = ent_end

    processed_text += text[current_ix:]
    return range_mapping, processed_text


class DeterministicIntentParser(IntentParser):
    unit_name = "deterministic_intent_parser"
    config_type = DeterministicIntentParserConfig

    def __init__(self, config=None):
        if config is None:
            config = self.config_type()
        super(DeterministicIntentParser, self).__init__(config)
        self.language = None
        self.regexes_per_intent = None
        self.group_names_to_slot_names = None
        self.slot_names_to_entities = None

    @property
    def patterns(self):
        if self.regexes_per_intent is not None:
            return {i: [r.pattern for r in regex_list] for i, regex_list in
                    iteritems(self.regexes_per_intent)}
        return None

    @patterns.setter
    def patterns(self, value):
        if value is not None:
            self.regexes_per_intent = dict()
            for intent, pattern_list in iteritems(value):
                regexes = [re.compile(r"%s" % p, re.IGNORECASE)
                           for p in pattern_list]
                self.regexes_per_intent[intent] = regexes

    @property
    def fitted(self):
        return self.regexes_per_intent is not None

    def fit(self, dataset, intents=None):
        self.language = Language.from_iso_code(dataset[LANGUAGE])
        self.regexes_per_intent = dict()
        self.group_names_to_slot_names = dict()
        joined_entity_utterances = get_joined_entity_utterances(
            dataset, self.language)
        self.slot_names_to_entities = get_slot_names_mapping(dataset)
        for intent_name, intent in iteritems(dataset[INTENTS]):
            if not self.is_trainable(intent, dataset):
                self.regexes_per_intent[intent_name] = []
                continue
            utterances = [preprocess_builtin_entities(u, self.language)
                          for u in intent[UTTERANCES]]
            regexes, self.group_names_to_slot_names = generate_regexes(
                utterances, joined_entity_utterances,
                self.group_names_to_slot_names, self.language)
            self.regexes_per_intent[intent_name] = regexes
        return self

    def is_trainable(self, intent, dataset):
        if len(intent[UTTERANCES]) >= self.config.max_queries:
            return False

        intent_entities = set(chunk[ENTITY] for query in intent[UTTERANCES]
                              for chunk in query[DATA] if ENTITY in chunk)
        total_entities = sum(len(dataset[ENTITIES][ent][UTTERANCES])
                             for ent in intent_entities
                             if not is_builtin_entity(ent))
        if total_entities > self.config.max_entities:
            return False
        return True

    def get_intent(self, text, intents=None):
        if not self.fitted:
            raise AssertionError("DeterministicIntentParser must be fitted "
                                 "before calling `get_intent`")
        return self._parse(text, intents)[RES_INTENT]

    def get_slots(self, text, intent):
        if not self.fitted:
            raise AssertionError("DeterministicIntentParser must be fitted "
                                 "before calling `get_slots`")
        if intent not in self.regexes_per_intent:
            raise KeyError("Intent not found in DeterministicIntentParser: %s"
                           % intent)
        return self._parse(text, [intent])[RES_SLOTS]

    def _parse(self, text, intents=None):
        ranges_mapping, processed_text = replace_builtin_entities(
            text, self.language)

        for intent, regexes in iteritems(self.regexes_per_intent):
            if intents is not None and intent not in intents:
                continue
            for regex in regexes:
                match = regex.match(processed_text)
                if match is None:
                    continue
                parsed_intent = intent_classification_result(
                    intent_name=intent, probability=1.0)
                slots = []
                for group_name in match.groupdict():
                    slot_name = self.group_names_to_slot_names[group_name]
                    entity = self.slot_names_to_entities[slot_name]
                    rng = (match.start(group_name), match.end(group_name))
                    value = match.group(group_name)
                    if rng in ranges_mapping:
                        rng = ranges_mapping[rng]
                        value = text[rng[0]:rng[1]]
                    parsed_slot = _slot(
                        match_range=rng, value=value, entity=entity,
                        slot_name=slot_name)
                    slots.append(parsed_slot)
                parsed_slots = deduplicate_overlapping_slots(
                    slots, self.language)
                parsed_slots = sorted(parsed_slots,
                                      key=lambda s: s[RES_MATCH_RANGE][0])
                return parsing_result(text, parsed_intent, parsed_slots)
        return empty_result(text)

    def to_dict(self):
        language_code = None
        if self.language is not None:
            language_code = self.language.iso_code
        return {
            "unit_name": self.unit_name,
            "config": self.config.to_dict(),
            "language_code": language_code,
            "patterns": self.patterns,
            "group_names_to_slot_names": self.group_names_to_slot_names,
            "slot_names_to_entities": self.slot_names_to_entities
        }

    @classmethod
    def from_dict(cls, unit_dict):
        config = cls.config_type.from_dict(unit_dict["config"])
        parser = cls(config=config)
        language = None
        if unit_dict["language_code"] is not None:
            language = Language.from_iso_code(unit_dict["language_code"])
        parser.patterns = unit_dict["patterns"]
        parser.language = language
        parser.group_names_to_slot_names = unit_dict[
            "group_names_to_slot_names"]
        parser.slot_names_to_entities = unit_dict["slot_names_to_entities"]
        return parser

    def __eq__(self, other):
        if not isinstance(other, DeterministicIntentParser):
            return False
        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        return not self.__eq__(other)
