import io
import json
import re

from custom_intent_parser.result import parsed_entity
from entity_extractor import EntityExtractor

GROUP_NAME_PREFIX = "group"
GROUP_NAME_SEPARATOR = "_"


def order(index):
    split = index.split(GROUP_NAME_SEPARATOR)
    if len(split) != 2 or split[0] != GROUP_NAME_PREFIX:
        raise ValueError("Misformatted index")
    return int(split[1])


def make_index(i):
    return "%s%s%s" % (GROUP_NAME_PREFIX, GROUP_NAME_SEPARATOR, i)


def generate_new_index(roles_to_labels):
    if len(roles_to_labels) == 0:
        return make_index(0)
    else:
        max_index = max(roles_to_labels.keys(), key=order)
        max_index = int(max_index.split(GROUP_NAME_SEPARATOR)[1]) + 1
        return make_index(max_index)


def query_to_pattern(query, joined_entity_utterances, roles_to_labels):
    pattern = r"^"
    for chunk in query["data"]:
        if "entity" in chunk:
            max_index = generate_new_index(roles_to_labels)
            role_name = chunk.get("role", None)
            label = (chunk["entity"], role_name)
            roles_to_labels[max_index] = label
            pattern += r"(?P<%s>%s)" % (
                max_index, joined_entity_utterances[chunk["entity"]])
        else:
            pattern += re.escape(chunk["text"])

    return pattern + r"$", roles_to_labels


def generate_regexes(intent_queries, entities, group_names_to_labels):
    # Join all the entities utterances with a "|" to create the patterns
    joined_entity_utterances = dict()
    for entity_name, entity in entities.iteritems():
        if entity.use_synonyms:
            utterances = [s for entry in entity.entries
                          for s in entry["synonyms"]]
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
            role=match.get("role", None), intent=match["intent"])
        results.append(parsed_ent)
    return results


class RegexEntityExtractor(EntityExtractor):
    def __init__(self, regexes=None, group_names_to_labels=None):
        """
        :param regexes: dict. The keys of the dict are (entity_name, role_name)
          pairs. role_name can be None if there is no role associated with the
          entity
        """
        if regexes is None:
            regexes = {}
        if group_names_to_labels is None:
            group_names_to_labels = {}
        self.regexes = regexes
        self.group_names_to_labels = group_names_to_labels

    @property
    def regexes(self):
        return self._regexes

    @regexes.setter
    def regexes(self, value):
        for pattern_list in value.values():
            for pattern in pattern_list:
                if not isinstance(pattern, re._pattern_type):
                    raise TypeError("patterns must be compile regexes "
                                    "founds %s" % type(pattern))
        self._regexes = value

    @property
    def group_names_to_labels(self):
        return self._group_names_to_labels

    @group_names_to_labels.setter
    def group_names_to_labels(self, value):
        for group_name, label in value.iteritems():
            try:
                eval(group_name)
            except NameError:
                pass
            except TypeError:
                raise TypeError("group names must be valid Python identifiers")

            if not isinstance(label, tuple) or len(label) != 2:
                raise TypeError("labels must be a dict with (entity_name,"
                                " role_name) 2-tuples as keys.")
        self._group_names_to_labels = value

    @property
    def fitted(self):
        return len(self.regexes) > 0

    def fit(self, dataset):
        # TODO: handle use_learning = True
        regexes = dict()
        group_names_to_labels = dict()
        for intent_name, intent_queries in dataset.queries.iteritems():
            intent_patterns, group_names_to_labels = generate_regexes(
                intent_queries, dataset.entities, group_names_to_labels)
            if len(intent_patterns) > 0:
                regexes[intent_name] = intent_patterns
        self.regexes = regexes
        self.group_names_to_labels = group_names_to_labels
        return self

    def get_entities(self, text):
        self.check_fitted()
        # Matches is a dict to ensure that we have only 1 match per range
        matches = dict()
        for intent_name, intent_regexes in self.regexes.iteritems():
            for regex in intent_regexes:
                match = regex.match(text)
                if match is not None:
                    for group_name in match.groupdict():
                        entity, role = self.group_names_to_labels[group_name]
                        rng = (match.start(group_name), match.end(group_name))
                        matches[rng] = {
                            "value": match.group(group_name),
                            "entity": entity,
                            "intent": intent_name,
                            "role": role
                        }
        return match_to_result(matches.items())

    def save(self, path):
        patterns = dict(self.regexes)
        for intent_name, intent_regexes in self.regexes.iteritems():
            patterns[intent_name] = [r.pattern for r in intent_regexes]
        self_as_dict = {
            "patterns": patterns,
            "group_names_to_labels": self.group_names_to_labels
        }
        with io.open(path, "w", encoding="utf8") as f:
            data = json.dumps(self_as_dict)
            f.write(unicode(data))

    @classmethod
    def load(cls, path):
        with io.open(path, encoding="utf8") as f:
            data = json.load(f)
        regexes = dict()
        for intent_name, patterns in data["patterns"].iteritems():
            regexes[intent_name] = [re.compile(r"%s" % p, re.IGNORECASE)
                                    for p in patterns]
        group_names_to_labels = data["group_names_to_labels"]
        for group_name, label in group_names_to_labels.iteritems():
            group_names_to_labels[group_name] = tuple(label)
        return cls(regexes, group_names_to_labels)
