import io
import json
import re

from snips_nlu.result import parsed_entity

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


def query_to_pattern(query, joined_entity_utterances, slots_names_to_labels):
    pattern = r"^"
    for chunk in query["data"]:
        if "entity" in chunk:
            max_index = generate_new_index(slots_names_to_labels)
            slot_name = chunk.get("slotName", None)
            label = (chunk["entity"], slot_name)
            slots_names_to_labels[max_index] = label
            pattern += r"(?P<%s>%s)" % (
                max_index, joined_entity_utterances[chunk["entity"]])
        else:
            pattern += re.escape(chunk["text"])

    return pattern + r"$", slots_names_to_labels


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
            slot_name=match.get("slotName", None), intent=match["intent"])
        results.append(parsed_ent)
    return results


class RegexEntityExtractor(object):
    def __init__(self, regexes=None, group_names_to_labels=None):
        """
        :param regexes: dict. The keys of the dict are (entity_name, slot_name)
          pairs. slot_name can be None if there is no slotName associated with
          the entity
        """
        self._regexes = None
        self.regexes = regexes
        self._group_names_to_labels = None
        self.group_names_to_labels = group_names_to_labels

    @property
    def regexes(self):
        return self._regexes

    @regexes.setter
    def regexes(self, value):
        if value is None:
            self._regexes = dict()
            return

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
        if value is None:
            self._group_names_to_labels = dict()
            return

        for group_name, label in value.iteritems():
            try:
                eval(group_name)
            except NameError:
                pass
            except TypeError:
                raise TypeError("group names must be valid Python identifiers")

            if not isinstance(label, tuple) or len(label) != 2:
                raise TypeError("labels must be a dict with (entity,"
                                " slotName) 2-tuples as keys.")
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
        if not self.fitted:
            raise ValueError("EntityExtractor must be fitted before calling "
                             "the 'get_entities' method.")

        # Matches is a dict to ensure that we have only 1 match per range
        matches = dict()
        for intent_name, intent_regexes in self.regexes.iteritems():
            for regex in intent_regexes:
                match = regex.match(text)
                if match is not None:
                    for group_name in match.groupdict():
                        entity, slot_name = self.group_names_to_labels[
                            group_name]
                        rng = (match.start(group_name), match.end(group_name))
                        matches[rng] = {
                            "value": match.group(group_name),
                            "entity": entity,
                            "intent": intent_name,
                            "slotName": slot_name
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
