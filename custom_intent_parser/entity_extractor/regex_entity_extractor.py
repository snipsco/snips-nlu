import re

from custom_intent_parser.result import parsed_entity
from entity_extractor import EntityExtractor


def query_to_pattern(query, target_entity_name, target_role,
                     joined_entity_utterances):
    pattern = r"^"
    for chunk in query["data"]:
        if "entity" in chunk:
            if chunk["entity"] == target_entity_name \
                    and chunk.get("role", None) == target_role:
                pattern += r"(%s)" % joined_entity_utterances[chunk["entity"]]
            else:
                pattern += r"(?:%s)" % joined_entity_utterances[
                    chunk["entity"]]
        else:
            pattern += r"%s" % chunk["text"]
    return pattern + r"$"


def generate_role_regexes(queries, target_entity, target_role,
                          joined_entity_utterances):
    # Filter queries by role
    role_queries = []
    for q in queries:
        if any(c.get("entity", None) == target_entity
               and c.get("role", None) == target_role for c in q["data"]):
            role_queries.append(q)
    # Extract the patterns for this role
    role_patterns = []
    for q in role_queries:
        pattern = query_to_pattern(q, target_entity, target_role,
                                   joined_entity_utterances)
        role_patterns.append(pattern)
    role_patterns = set(role_patterns)
    return [re.compile(p, re.IGNORECASE) for p in role_patterns]


def generate_regexes(intent_queries, target_entity, entities):
    # Join all the entities utterances with a "|" to create the patterns
    joined_entity_utterances = dict()
    for entity_name, entity in entities.iteritems():
        if entity.use_synonyms:
            utterances = [re.escape(s) for entry in entity.entries
                          for s in entry["synonyms"]]
        else:
            utterances = [re.escape(entry["value"])
                          for entry in entity.entries]
        joined_entity_utterances[entity_name] = r"|".join(
            utterances)

    # Filter queries by entity type
    queries = []
    for q in intent_queries:
        if any(c.get("entity", None) == target_entity.name for c in q["data"]):
            queries.append(q)

    # Extract the different roles
    roles = set()
    for q in queries:
        for chunk in q["data"]:
            roles.add(chunk.get("role", None))

    # Filter queries by role
    patterns = dict()
    for role in roles:
        role_regexes = generate_role_regexes(
            queries, target_entity.name, role, joined_entity_utterances)
        if len(role_regexes) > 0:
            patterns[(target_entity.name, role)] = role_regexes
    return patterns


def match_to_result(matches):
    results = []
    for match_rng, match in matches:
        parsed_ent = parsed_entity(
            match_rng, match["value"], match["entity"],
            role=match.get("role", None), intent=match["intent"])
        results.append(parsed_ent)
    return results


class RegexEntityExtractor(EntityExtractor):
    def __init__(self, regexes=None):
        """
        :param regexes: dict. The keys of the dict are (entity_name, role_name)
          pairs. role_name can be None if there is no role associated with the
          entity
        """
        if regexes is None:
            regexes = {}
        self.regexes = regexes

    @property
    def regexes(self):
        return self._regexes

    @regexes.setter
    def regexes(self, value):
        for intent_name, intent_patterns in value.iteritems():
            for k, pattern_list in intent_patterns.iteritems():
                if not isinstance(k, tuple) or len(k) != 2:
                    raise TypeError("regexes must be a dict with (entity_name,"
                                    " role_name) 2-tuples as keys.")
                for pattern in pattern_list:
                    if not isinstance(pattern, re._pattern_type):
                        raise TypeError("patterns must be compile regexes "
                                        "founds %s" % type(pattern))
        self._regexes = value

    @property
    def fitted(self):
        return len(self.regexes) > 0

    def fit(self, dataset):
        # TODO: handle use_learning = True
        regexes = dict()
        for intent_name, intent_queries in dataset.queries.iteritems():
            regexes[intent_name] = dict()
            for target_name, target_entity in dataset.entities.iteritems():
                patterns_dict = generate_regexes(intent_queries, target_entity,
                                                 dataset.entities)
                for (entity_name, role), pattern in patterns_dict.iteritems():
                    regexes[intent_name][(entity_name, role)] = pattern
        self.regexes = regexes
        return self

    def get_entities(self, text):
        self.check_fitted()
        # Matches is a dict to ensure that we have only 1 match per range
        matches = dict()
        for intent_name in self.regexes:
            for (entity_name, role), entity_regexes \
                    in self.regexes[intent_name].iteritems():
                for regex in entity_regexes:
                    for match in regex.finditer(text):
                        match_range = match.regs[1]
                        if match_range not in matches:
                            matches[match_range] = {
                                "value": match.group(1),
                                "entity": entity_name,
                                "intent": intent_name,
                                "role": role
                            }
        return match_to_result(matches.items())
