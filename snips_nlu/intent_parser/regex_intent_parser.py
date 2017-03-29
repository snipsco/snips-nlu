import operator
import re

from snips_nlu.constants import (TEXT, USE_SYNONYMS, SYNONYMS, DATA, INTENTS,
                                 ENTITIES, SLOT_NAME, UTTERANCES, VALUE,
                                 ENTITY)
from snips_nlu.built_in_entities import BuiltInEntity
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.result import (IntentClassificationResult,
                              ParsedSlot)
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import instance_to_generic_dict

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
                     group_names_to_slot_names):
    pattern = r"^"
    for chunk in query[DATA]:
        if SLOT_NAME in chunk and chunk[ENTITY] not in \
                BuiltInEntity.built_in_entity_by_label:
            max_index = generate_new_index(group_names_to_slot_names)
            slot_name = chunk[SLOT_NAME]
            entity = chunk[ENTITY]
            group_names_to_slot_names[max_index] = slot_name
            pattern += r"(?P<%s>%s)" % (
                max_index, joined_entity_utterances[entity])
        else:
            pattern += re.escape(chunk[TEXT])

    return pattern + r"$", group_names_to_slot_names


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
        if entity[USE_SYNONYMS]:
            utterances = [syn for entry in entity[DATA]
                          for syn in entry[SYNONYMS]]
        else:
            utterances = [entry[VALUE] for entry in entity[DATA]]
        joined_entity_utterances[entity_name] = r"|".join(
            sorted([re.escape(e) for e in utterances], key=len, reverse=True))
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


class RegexIntentParser(IntentParser):
    def __init__(self, patterns=None, group_names_to_slot_names=None,
                 slot_names_to_entities=None):
        self.regexes_per_intent = None
        if patterns is not None:
            self.regexes_per_intent = dict()
            for intent, patterns in patterns.iteritems():
                regexes = [re.compile(r"%s" % p, re.IGNORECASE) for p in
                           patterns]
                self.regexes_per_intent[intent] = regexes
        self.group_names_to_slot_names = group_names_to_slot_names
        self.slot_names_to_entities = slot_names_to_entities

    @property
    def fitted(self):
        return self.regexes_per_intent is not None

    def fit(self, dataset):
        self.regexes_per_intent = dict()
        self.group_names_to_slot_names = dict()
        joined_entity_utterances = get_joined_entity_utterances(dataset)
        self.slot_names_to_entities = get_slot_names_mapping(dataset)
        for intent_name, intent in dataset[INTENTS].iteritems():
            utterances = intent[UTTERANCES]
            regexes, self.group_names_to_slot_names = generate_regexes(
                utterances, joined_entity_utterances,
                self.group_names_to_slot_names)
            self.regexes_per_intent[intent_name] = regexes
        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError("RegexIntentParser must be fitted before "
                                 "calling `get_entities`")
        entities_per_intent = dict()
        for intent in self.regexes_per_intent.keys():
            entities_per_intent[intent] = self.get_slots(text, intent)

        intents_probas = dict()
        total_nb_entities = sum(
            len(entities) for entities in entities_per_intent.values())
        # TODO: handle intents without slots
        if total_nb_entities == 0:
            return None
        for intent_name, entities in entities_per_intent.iteritems():
            intents_probas[intent_name] = float(len(entities)) / float(
                total_nb_entities)

        top_intent, top_proba = max(intents_probas.items(),
                                    key=operator.itemgetter(1))
        return IntentClassificationResult(intent_name=top_intent,
                                          probability=top_proba)

    def get_slots(self, text, intent=None):
        if not self.fitted:
            raise AssertionError("RegexIntentParser must be fitted before "
                                 "calling `get_entities`")
        if intent not in self.regexes_per_intent:
            raise KeyError("Intent not found in RegexIntentParser: %s"
                           % intent)
        slots = []
        for regex in self.regexes_per_intent[intent]:
            match = regex.match(text)
            if match is None:
                continue
            for group_name in match.groupdict():
                slot_name = self.group_names_to_slot_names[group_name]
                entity = self.slot_names_to_entities[slot_name]
                rng = (match.start(group_name), match.end(group_name))
                parsed_slot = ParsedSlot(match_range=rng,
                                         value=match.group(group_name),
                                         entity=entity,
                                         slot_name=slot_name)
                slots.append(parsed_slot)
        return deduplicate_overlapping_slots(slots)

    def to_dict(self):
        obj_dict = instance_to_generic_dict(self)
        if self.regexes_per_intent is not None:
            patterns = {i: [r.pattern for r in regex_list] for i, regex_list in
                        self.regexes_per_intent.iteritems()}
        else:
            patterns = None

        obj_dict.update({
            "patterns": patterns,
            "group_names_to_slot_names": self.group_names_to_slot_names,
            "slot_names_to_entities": self.slot_names_to_entities
        })

        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        patterns = obj_dict["patterns"]
        group_names_to_slot_names = obj_dict["group_names_to_slot_names"]
        slot_names_to_entities = obj_dict["slot_names_to_entities"]
        return cls(patterns, group_names_to_slot_names,
                   slot_names_to_entities)
