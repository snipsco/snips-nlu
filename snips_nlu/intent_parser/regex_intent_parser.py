import io
import json
import re

from snips_nlu.result import result, intent_classification_result, parsed_entity
from ..intent_parser.intent_parser import IntentParser, CUSTOM_PARSER_TYPE

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


def get_slot_names_mapping(intent_queries):
    slot_names_to_entities = dict()
    for query in intent_queries:
        for chunk in query["data"]:
            if "entity" in chunk:
                slot_name = chunk["slotName"]
                entity = chunk["entity"]
                slot_names_to_entities[slot_name] = entity
    return slot_names_to_entities


def query_to_pattern(query, joined_entity_utterances,
                     group_names_to_slot_names):
    pattern = r"^"
    for chunk in query["data"]:
        if "entity" in chunk:
            max_index = generate_new_index(group_names_to_slot_names)
            slot_name = chunk["slotName"]
            entity = chunk["entity"]
            group_names_to_slot_names[max_index] = slot_name
            pattern += r"(?P<%s>%s)" % (
                max_index, joined_entity_utterances[entity])
        else:
            pattern += re.escape(chunk["text"])

    return pattern + r"$", group_names_to_slot_names


def generate_regexes(intent_queries, entities):
    # Join all the entities utterances with a "|" to create the patterns
    group_names_to_labels = dict()
    joined_entity_utterances = dict()
    for entity_name, entity in entities.iteritems():
        if entity.use_synonyms:
            utterances = [syn for entry in entity.entries
                          for syn in entry["synonyms"]]
        else:
            utterances = [entry["value"] for entry in entity.entries]
        joined_entity_utterances[entity_name] = r"|".join(
            sorted([re.escape(e) for e in utterances], key=len, reverse=True))
    patterns = set()
    for query in intent_queries:
        pattern, group_names_to_labels = query_to_pattern(
            query, joined_entity_utterances, group_names_to_labels)
        patterns.add(pattern)
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    return regexes, group_names_to_labels


def match_to_result(matches):
    results = []
    for match_rng, match in matches:
        parsed_ent = parsed_entity(
            match_rng, match["value"], match["entity"],
            slot_name=match.get("slotName", None))
        results.append(parsed_ent)
    results.sort(key=lambda res: res["range"][0])
    return results


class RegexIntentParser(IntentParser):
    def __init__(self, intent_name, patterns=None,
                 group_names_to_slot_names=None, slot_names_to_entities=None):
        super(RegexIntentParser, self).__init__(intent_name,
                                                CUSTOM_PARSER_TYPE)
        self.regexes = None
        if patterns is not None:
            self.regexes = [re.compile(r"%s" % p, re.IGNORECASE) for p in
                            patterns]
        self.group_names_to_slot_names = group_names_to_slot_names
        self.slot_names_to_entities = slot_names_to_entities

    @property
    def fitted(self):
        return self.regexes is not None

    def fit(self, dataset):
        if self.intent_name not in dataset.queries:
            return self

        intent_queries = dataset.queries[self.intent_name]
        self.regexes, self.group_names_to_slot_names = generate_regexes(
            intent_queries, dataset.entities)
        self.slot_names_to_entities = get_slot_names_mapping(intent_queries)
        return self

    def parse(self, text):
        if len(text) == 0:
            return result(text)
        entities = self.get_entities(text)
        entities_length = 0
        for entity in entities:
            entities_length += entity["range"][1] - entity["range"][0]
        intent_score = entities_length / float(len(text))
        intent_result = intent_classification_result(self.intent_name,
                                                     intent_score)
        return result(text, intent_result, entities)

    def get_intent(self, text):
        if len(text) == 0:
            return result(text)
        entities = self.get_entities(text)
        entities_length = 0
        for entity in entities:
            entities_length += entity["range"][1] - entity["range"][0]
        intent_score = entities_length / float(len(text))
        return intent_classification_result(self.intent_name, intent_score)

    def get_entities(self, text, intent=None):
        if not self.fitted:
            raise ValueError("Parser must be fitted before calling the "
                             "'get_entities' method.")

        # Matches is a dict to ensure that we have only 1 match per range
        matches = dict()
        for regex in self.regexes:
            match = regex.match(text)
            if match is not None:
                for group_name in match.groupdict():
                    slot_name = self.group_names_to_slot_names[
                        group_name]
                    entity = self.slot_names_to_entities[slot_name]
                    rng = (match.start(group_name), match.end(group_name))
                    matches[rng] = {
                        "value": match.group(group_name),
                        "entity": entity,
                        "slotName": slot_name
                    }
        return match_to_result(matches.items())

    def save(self, path):
        self_as_dict = {
            "intent_name": self.intent_name,
            "patterns": [r.pattern for r in self.regexes],
            "group_names_to_slot_names": self.group_names_to_slot_names,
            "slot_names_to_entities": self.slot_names_to_entities,
        }
        with io.open(path, "w", encoding="utf8") as f:
            data = json.dumps(self_as_dict)
            f.write(unicode(data))

    @classmethod
    def load(cls, path):
        with io.open(path, encoding="utf8") as f:
            data = json.load(f)
        return cls(
            data["intent_name"],
            data["patterns"],
            data["group_names_to_slot_names"],
            data["slot_names_to_entities"]
        )

    def __eq__(self, other):
        if self.intent_name != other.intent_name:
            return False
        if self.group_names_to_slot_names != other.group_names_to_slot_names:
            return False
        if self.slot_names_to_entities != other.slot_names_to_entities:
            return False
        if self.regexes is not None and other.regexes is not None:
            self_patterns = [r.pattern for r in self.regexes]
            self_flags = [r.flags for r in self.regexes]
            other_patterns = [r.pattern for r in other.regexes]
            other_flags = [r.flags for r in other.regexes]
            if self_patterns != other_patterns or self_flags != other_flags:
                return False
        if self.regexes is None and other.regexes is not None:
            return False
        if self.regexes is not None and other.regexes is None:
            return False

        return True
