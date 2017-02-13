import re
from collections import OrderedDict

from custom_intent_parser.entity_extractor.entity_extractor import \
    EntityExtractor


def regex_for_first_chunk(current_chunk, right_context_chunks):
    right_context = []
    string_pattern = None
    token_regex = re.compile(r"(\w+)", re.IGNORECASE)
    num_tokens = len(token_regex.findall(current_chunk["text"]))

    if len(right_context_chunks) == 0:
        string_pattern = r"(%s)" % current_chunk["text"]

    for i, chunk in enumerate(right_context_chunks):
        if "entity" in chunk or i == len(right_context_chunks):
            if i == 0:
                string_pattern = r"(%s)" % current_chunk["text"]
            else:
                right_context = "".join(right_context)
                inside = r"\s*".join(r"\w+\b" for _ in xrange(num_tokens))
                string_pattern = r"(%s)%s" % (inside, "".join(right_context))
                break
        else:
            right_context.append(chunk["text"])

    # If we went out of the loop before breaking
    if string_pattern is None:
        inside = r"\s*".join(r"\w+\b" for _ in xrange(num_tokens))
        string_pattern = r"(%s)%s" % (inside, "".join(right_context))

    regex = re.compile(string_pattern, re.IGNORECASE)
    assert isinstance(regex, re._pattern_type)
    return regex


def regex_from_left_context(current_chunk, left_context_chunks):
    left_context = []
    string_pattern = None
    token_regex = re.compile(r"(\w+)", re.IGNORECASE)
    num_tokens = len(token_regex.findall(current_chunk["text"]))

    if len(left_context_chunks) == 0:
        string_pattern = r"(%s)" % current_chunk["text"]
    for i, chunk in enumerate(reversed(left_context_chunks)):
        if "entity" in chunk or i == len(left_context_chunks):
            if i == 0:
                string_pattern = r"(%s)" % current_chunk["text"]
            else:
                left_context = "".join(reversed(left_context))
                inside = r"\s*".join(r"\w+\b" for _ in xrange(num_tokens))
                string_pattern = r"%s(%s)" % ("".join(left_context), inside)
                break
        else:
            left_context.append(chunk["text"])

    # If we went out of the loop before breaking
    if string_pattern is None:
        inside = r"\s*".join(r"\w+\b" for _ in xrange(num_tokens))
        string_pattern = r"%s(%s)" % ("".join(left_context), inside)
    regex = re.compile(string_pattern, re.IGNORECASE)
    assert isinstance(regex, re._pattern_type)
    return regex


def query_to_patterns(query):
    patterns = dict()
    data = query["data"]
    left_context = ""
    for i, chunk in enumerate(data):
        if "entity" in chunk:
            entity_name = chunk["entity"]
            if entity_name not in patterns:
                patterns[entity_name] = []
            if i == 0:
                if len(data) == 1:
                    regex = re.compile(r"^(%s)$" % data["text"], re.IGNORECASE)
                else:
                    regex = regex_for_first_chunk(chunk, data[i + 1:])
            else:
                regex = regex_from_left_context(chunk, data[0:i])

            patterns[entity_name].append(regex)
            left_context = ""
        else:
            left_context += chunk["text"]

    return patterns


class RegexEntityExtractor(EntityExtractor):
    # TODO: handle role
    def __init__(self, regexes=None):
        if regexes is None:
            regexes = OrderedDict()
        if not isinstance(regexes, OrderedDict):
            raise ValueError("regexes must be an OrderedDict, found %s. "
                             "The first matching regex will give its label "
                             "to the entity" % type(regexes))
        self.regexes = regexes

    @property
    def fitted(self):
        return len(self.regexes) > 0

    @property
    def entities(self):
        return self.regexes.keys()

    def fit(self, dataset):
        queries = dataset.queries
        # TODO: modify the pattern extract to have entity patterns by intent
        regexes = OrderedDict()
        for _, intent_queries in queries.iteritems():
            for query in intent_queries:
                patterns = query_to_patterns(query)
                for entity_name, entity_patterns in patterns.iteritems():
                    if entity_name not in regexes:
                        regexes[entity_name] = entity_patterns
                    else:
                        regexes[entity_name] += entity_patterns
        self.regexes = regexes
        return self

    def get_entities(self, text):
        self.check_fitted()
        matches = dict()
        for entity_name, entity_regexes in self.regexes.iteritems():
            for regex in entity_regexes:
                for match in regex.finditer(text):
                    match_range = match.regs[1]
                    if match_range not in matches:
                        matches[match_range] = {
                            "value": match.group(1),
                            "entity": entity_name
                        }
        return self.match_to_result(matches.items())

    @staticmethod
    def match_to_result(matches):
        results = []
        for match_range, match in matches:
            entity = {
                "range": match_range,
                "value": match["value"],
                "entity": match["entity"]
            }
            results.append(entity)
        return results

    @classmethod
    def from_dataset(cls, dataset):
        extractor = cls()
        extractor.fit(dataset.queries)
        return extractor
