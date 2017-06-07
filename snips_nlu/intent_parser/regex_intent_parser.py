from __future__ import unicode_literals

import re
from copy import deepcopy

from snips_nlu.builtin_entities import is_builtin_entity, \
    get_builtin_entities
from snips_nlu.constants import (TEXT, USE_SYNONYMS, SYNONYMS, DATA, INTENTS,
                                 ENTITIES, SLOT_NAME, UTTERANCES, VALUE,
                                 ENTITY, MATCH_RANGE)
from snips_nlu.languages import Language
from snips_nlu.result import (IntentClassificationResult,
                              ParsedSlot, Result)
from snips_nlu.tokenization import tokenize, tokenize_light
from snips_nlu.utils import LimitedSizeDict, regex_escape

GROUP_NAME_PREFIX = "group"
GROUP_NAME_SEPARATOR = "_"
SPACE = " "
WHITE_SPACES = "%s\t\n\r\f\v" % SPACE  # equivalent of r"\s"
IGNORED_CHARACTERS = "%s.,;/:+*-`\"(){}" % WHITE_SPACES
IGNORED_CHARACTERS_PATTERN = r"[%s]*" % IGNORED_CHARACTERS


def get_index(index):
    split = index.split(GROUP_NAME_SEPARATOR)
    if len(split) != 2 or split[0] != GROUP_NAME_PREFIX:
        raise ValueError("Misformatted index")
    return int(split[1])


def make_index(i):
    return "%s%s%s" % (GROUP_NAME_PREFIX, GROUP_NAME_SEPARATOR, i)


def generate_new_index(slots_name_to_labels):
    if len(slots_name_to_labels) == 0:
        return make_index(0)
    else:
        max_index = max(slots_name_to_labels.keys(), key=get_index)
        max_index = get_index(max_index) + 1
        return make_index(max_index)


def get_slot_names_mapping(dataset):
    slot_names_to_entities = dict()
    for intent in dataset[INTENTS].values():
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if SLOT_NAME in chunk:
                    slot_name = chunk[SLOT_NAME]
                    entity = chunk[ENTITY]
                    slot_names_to_entities[slot_name] = entity
    return slot_names_to_entities


def query_to_pattern(query, joined_entity_utterances,
                     group_names_to_slot_names, ):
    pattern = []
    for i, chunk in enumerate(query[DATA]):
        if SLOT_NAME in chunk:
            max_index = generate_new_index(group_names_to_slot_names)
            slot_name = chunk[SLOT_NAME]
            entity = chunk[ENTITY]
            group_names_to_slot_names[max_index] = slot_name
            pattern.append(
                r"(?P<%s>%s)" % (max_index, joined_entity_utterances[entity]))
        else:
            tokens = tokenize_light(chunk[TEXT])
            pattern += [regex_escape(t) for t in tokens]
    pattern = r"^%s%s%s$" % (IGNORED_CHARACTERS_PATTERN,
                             IGNORED_CHARACTERS_PATTERN.join(pattern),
                             IGNORED_CHARACTERS_PATTERN)
    return pattern, group_names_to_slot_names


def generate_regexes(intent_queries, joined_entity_utterances,
                     group_names_to_labels):
    # Join all the entities utterances with a "|" to create the patterns
    patterns = set()
    for query in intent_queries:
        pattern, group_names_to_labels = query_to_pattern(
            query, joined_entity_utterances, group_names_to_labels)
        patterns.add(pattern)
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    return regexes, group_names_to_labels


def get_joined_entity_utterances(dataset):
    joined_entity_utterances = dict()
    for entity_name, entity in dataset[ENTITIES].iteritems():
        if is_builtin_entity(entity_name):
            utterances = [get_builtin_entity_name(entity_name)]
        else:
            if entity[USE_SYNONYMS]:
                utterances = [syn for entry in entity[DATA]
                              for syn in entry[SYNONYMS]]
            else:
                utterances = [entry[VALUE] for entry in entity[DATA]]
        utterances_patterns = [regex_escape(e) for e in utterances]
        joined_entity_utterances[entity_name] = r"|".join(
            sorted(utterances_patterns, key=len, reverse=True))
    return joined_entity_utterances


def deduplicate_overlapping_slots(slots):
    deduplicated_slots = []
    for slot in slots:
        is_overlapping = False
        for slot_index, dedup_slot in enumerate(deduplicated_slots):
            if slot.match_range[1] > dedup_slot.match_range[0] \
                    and slot.match_range[0] < dedup_slot.match_range[1]:
                is_overlapping = True
                tokens = tokenize(slot.value)
                dedup_tokens = tokenize(dedup_slot.value)
                if len(tokens) > len(dedup_tokens):
                    deduplicated_slots[slot_index] = slot
                elif len(tokens) == len(dedup_tokens) \
                        and len(slot.value) > len(dedup_slot.value):
                    deduplicated_slots[slot_index] = slot
        if not is_overlapping:
            deduplicated_slots.append(slot)
    return deduplicated_slots


def get_builtin_entity_name(entity_label):
    return "%%%s%%" % "".join(tokenize_light(entity_label)).upper()


def preprocess_builtin_entities(utterance):
    new_utterance = deepcopy(utterance)
    for i, chunk in enumerate(utterance[DATA]):
        if ENTITY in chunk and is_builtin_entity(chunk[ENTITY]):
            new_utterance[DATA][i][TEXT] = get_builtin_entity_name(
                chunk[ENTITY])
    return new_utterance


def replace_builtin_entities(text, language):
    builtin_entities = get_builtin_entities(text, language)
    if len(builtin_entities) == 0:
        return dict(), text

    range_mapping = dict()
    processed_text = ""
    offset = 0
    current_ix = 0
    builtin_entities = sorted(builtin_entities,
                              key=lambda e: e[MATCH_RANGE][0])
    for ent in builtin_entities:
        ent_start = ent[MATCH_RANGE][0]
        ent_end = ent[MATCH_RANGE][1]
        rng_start = ent_start + offset

        processed_text += text[current_ix:ent_start]

        entity_text = get_builtin_entity_name(ent[ENTITY].label)
        offset += len(entity_text) - (
            ent[MATCH_RANGE][1] - ent[MATCH_RANGE][0])

        processed_text += entity_text
        rng_end = ent_end + offset
        new_range = (rng_start, rng_end)
        range_mapping[new_range] = ent[MATCH_RANGE]
        current_ix = ent_end

    processed_text += text[current_ix:]
    return range_mapping, processed_text


class RegexIntentParser(object):
    def __init__(self, language, patterns=None, group_names_to_slot_names=None,
                 slot_names_to_entities=None):
        self.language = language
        self.regexes_per_intent = None
        if patterns is not None:
            self.regexes_per_intent = dict()
            for intent, patterns in patterns.iteritems():
                regexes = [re.compile(r"%s" % p, re.IGNORECASE) for p in
                           patterns]
                self.regexes_per_intent[intent] = regexes
        self.group_names_to_slot_names = group_names_to_slot_names
        self.slot_names_to_entities = slot_names_to_entities
        self._cache = LimitedSizeDict(size_limit=1000)

    @property
    def fitted(self):
        return self.regexes_per_intent is not None

    def fit(self, dataset, intents=None):
        if intents is None:
            intents_to_train = dataset[INTENTS].keys()
        else:
            intents_to_train = list(intents)
        self.regexes_per_intent = dict()
        self.group_names_to_slot_names = dict()
        joined_entity_utterances = get_joined_entity_utterances(dataset)
        self.slot_names_to_entities = get_slot_names_mapping(dataset)
        for intent_name, intent in dataset[INTENTS].iteritems():
            if intent_name not in intents_to_train:
                self.regexes_per_intent[intent_name] = []
                continue
            utterances = [preprocess_builtin_entities(u)
                          for u in intent[UTTERANCES]]
            regexes, self.group_names_to_slot_names = generate_regexes(
                utterances, joined_entity_utterances,
                self.group_names_to_slot_names)
            self.regexes_per_intent[intent_name] = regexes
        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError("RegexIntentParser must be fitted before "
                                 "calling `get_entities`")
        if text not in self._cache:
            self._cache[text] = self._parse(text)
        return self._cache[text].parsed_intent

    def get_slots(self, text, intent=None):
        if not self.fitted:
            raise AssertionError("RegexIntentParser must be fitted before "
                                 "calling `get_entities`")
        if intent not in self.regexes_per_intent:
            raise KeyError("Intent not found in RegexIntentParser: %s"
                           % intent)
        if text not in self._cache:
            self._cache[text] = self._parse(text)
        res = self._cache[text]
        if intent is not None and res.parsed_intent is not None \
                and res.parsed_intent.intent_name != intent:
            return []
        return res.parsed_slots

    def _parse(self, text):
        if not self.fitted:
            raise AssertionError("RegexIntentParser must be fitted before "
                                 "calling `get_entities`")
        ranges_mapping, processed_text = replace_builtin_entities(
            text, self.language)

        parsed_intent = None
        parsed_slots = []
        matched = False
        for intent, regexes in self.regexes_per_intent.iteritems():
            for regex in regexes:
                match = regex.match(processed_text)
                if match is None:
                    continue
                parsed_intent = IntentClassificationResult(
                    intent_name=intent, probability=1.0)
                matched = True
                slots = []
                for group_name in match.groupdict():
                    slot_name = self.group_names_to_slot_names[group_name]
                    entity = self.slot_names_to_entities[slot_name]
                    rng = (match.start(group_name), match.end(group_name))
                    value = match.group(group_name)
                    if rng in ranges_mapping:
                        rng = ranges_mapping[rng]
                        value = text[rng[0]:rng[1]]
                    parsed_slot = ParsedSlot(
                        match_range=rng, value=value, entity=entity,
                        slot_name=slot_name)
                    slots.append(parsed_slot)
                parsed_slots = deduplicate_overlapping_slots(slots)
                break
            if matched:
                break
        return Result(text, parsed_intent, parsed_slots)

    def to_dict(self):
        patterns = None
        if self.regexes_per_intent is not None:
            patterns = {i: [r.pattern for r in regex_list] for i, regex_list in
                        self.regexes_per_intent.iteritems()}
        return {
            "language": self.language.iso_code,
            "patterns": patterns,
            "group_names_to_slot_names": self.group_names_to_slot_names,
            "slot_names_to_entities": self.slot_names_to_entities
        }

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict["language"])
        patterns = obj_dict["patterns"]
        group_names_to_slot_names = obj_dict["group_names_to_slot_names"]
        slot_names_to_entities = obj_dict["slot_names_to_entities"]
        return cls(language, patterns, group_names_to_slot_names,
                   slot_names_to_entities)

    def __eq__(self, other):
        if not isinstance(other, RegexIntentParser):
            return False
        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        return not self.__eq__(other)
