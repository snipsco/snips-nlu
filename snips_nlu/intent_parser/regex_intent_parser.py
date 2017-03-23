import operator
import re

from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.result import IntentClassificationResult, \
    ParsedSlot

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
    for intent in dataset["intents"].values():
        for utterance in intent["utterances"]:
            for chunk in utterance["data"]:
                if "entity" in chunk:
                    slot_name = chunk["slot_name"]
                    entity = chunk["entity"]
                    slot_names_to_entities[slot_name] = entity
    return slot_names_to_entities


def query_to_pattern(query, joined_entity_utterances,
                     group_names_to_slot_names):
    pattern = r"^"
    for chunk in query["data"]:
        if "entity" in chunk:
            max_index = generate_new_index(group_names_to_slot_names)
            slot_name = chunk["slot_name"]
            entity = chunk["entity"]
            group_names_to_slot_names[max_index] = slot_name
            pattern += r"(?P<%s>%s)" % (
                max_index, joined_entity_utterances[entity])
        else:
            pattern += re.escape(chunk["text"])

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
    for entity_name, entity in dataset["entities"].iteritems():
        if entity["use_synonyms"]:
            utterances = [syn for entry in entity["data"]
                          for syn in entry["synonyms"]]
        else:
            utterances = [entry["value"] for entry in entity["data"]]
        joined_entity_utterances[entity_name] = r"|".join(
            sorted([re.escape(e) for e in utterances], key=len, reverse=True))
    return joined_entity_utterances


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
        for intent_name, intent in dataset["intents"].iteritems():
            utterances = intent["utterances"]
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
        entities = []
        for regex in self.regexes_per_intent[intent]:
            match = regex.match(text)
            if match is None:
                continue
            for group_name in match.groupdict():
                slot_name = self.group_names_to_slot_names[group_name]
                entity = self.slot_names_to_entities[slot_name]
                rng = (match.start(group_name), match.end(group_name))
                parsed_entity = ParsedSlot(match_range=rng,
                                           value=match.group(group_name),
                                           entity=entity,
                                           slot_name=slot_name)
                entities.append(parsed_entity)
        return entities

    def __eq__(self, other):
        return isinstance(other, RegexIntentParser) and \
               self.group_names_to_slot_names \
               == other.group_names_to_slot_names and \
               self.slot_names_to_entities == other.slot_names_to_entities and \
               self.regexes_per_intent == other.regexes_per_intent

    def __ne__(self, other):
        return not self.__eq__(other)
