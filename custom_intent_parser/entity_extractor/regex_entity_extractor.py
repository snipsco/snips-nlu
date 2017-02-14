import re

from entity_extractor import EntityExtractor


def query_to_pattern(query, target_entity, joined_entity_utterances):
    pattern = r"^"
    for chunk in query["data"]:
        if "entity" in chunk:
            if chunk["entity"] == target_entity.name:
                pattern += r"(%s)" % joined_entity_utterances[chunk["entity"]]
            else:
                pattern += r"(?:%s)" % joined_entity_utterances[
                    chunk["entity"]]
        else:
            pattern += r"%s" % chunk["text"]
    return pattern + r"$"


def generate_entity_regexes(intent_queries, target_entity, entities):
    # Filter queries
    queries = [q for q in intent_queries
               if any(c.get("entity", None) == target_entity.name
                      for c in q["data"])]
    joined_entity_utterances = dict()
    for entity_name, entity in entities.iteritems():
        if entity.use_synonyms:
            utterances = [re.escape(s) for entry in entity.entries
                          for s in entry["synonyms"]]
        else:
            utterances = [re.escape(entry["value"])
                          for entry in entity.entries]
        joined_entity_utterances[entity_name] = r"|".join(utterances)

    patterns = []
    for q in queries:
        patterns.append(
            query_to_pattern(q, target_entity, joined_entity_utterances))
    patterns = set(patterns)
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class RegexEntityExtractor(EntityExtractor):
    def __init__(self, regexes=None):
        if regexes is None:
            regexes = {}
        assert isinstance(regexes, dict)
        self.regexes = regexes

    @property
    def fitted(self):
        return len(self.regexes) > 0

    def fit(self, dataset):
        # TODO: handle use_learning = True
        regexes = dict()
        for intent_name, intent_queries in dataset.queries.iteritems():
            regexes[intent_name] = dict()
            for target_name, target_entity in dataset.entities.iteritems():
                entity_patterns = generate_entity_regexes(
                    intent_queries, target_entity, dataset.entities)
                if len(entity_patterns) > 0:
                    regexes[intent_name][target_name] = entity_patterns

        self.regexes = regexes
        return self

    def get_entities(self, text):
        # TODO: handle roles
        self.check_fitted()
        matches = dict()
        for intent_name in self.regexes:
            for entity_name, entity_regexes \
                    in self.regexes[intent_name].iteritems():
                for regex in entity_regexes:
                    for match in regex.finditer(text):
                        match_range = match.regs[1]
                        if match_range not in matches:
                            matches[match_range] = {
                                "value": match.group(1),
                                "entity": entity_name,
                                "intent": intent_name
                            }
        return self.match_to_result(matches.items())

    @staticmethod
    def match_to_result(matches):
        results = []
        for match_range, match in matches:
            entity = {
                "range": match_range,
                "value": match["value"],
                "entity": match["entity"],
                "intent": match["intent"],
            }
            results.append(entity)
        return results
