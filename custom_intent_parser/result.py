from utils import namedtuple_with_defaults

_IntentClassificationResult = namedtuple_with_defaults(
    "_IntentClassificationResult", "name prob")


class IntentClassificationResult(_IntentClassificationResult):
    def to_dict(self):
        d = {}
        if self.intent is not None:
            d["intent"] = self.name
            if self.intent_prob is not None:
                d["prob"] = self.prob
        return d


class ParsedEntity(object):
    def __init__(self, match_range, value, entity, role=None, **kwargs):
        self._range = None
        self.range = match_range
        self.value = value
        self.entity = entity
        self.role = role

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("range must be a length 2 list or tuple")
        self._range = list(value)

    def to_dict(self):
        d = {
            "range": self.range,
            "value": self.value,
            "entity": self.entity
        }
        if self.role is not None:
            d["role"] = self.role
        return d


class EntityExtractionResult(object):
    def __init__(self, entities=None):
        self._entities = []
        self.entities = entities

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, values):
        if not isinstance(values, (list, tuple)):
            raise ValueError("entities must be a list or tuple of ParsedEntity"
                             " instance")
        for e in values:
            if not isinstance(e, ParsedEntity):
                raise ValueError("entities are expected to be ParsedEntity "
                                 "found %s with type %s" % (e, type(e)))
        self._entities = list(values)

    def as_list(self):
        return [e.to_dict() for e in self.entities]


class Result(object):
    def __init__(self, text, intent_result=None, entity_results=None):
        self.text = text
        self._intent_result = None
        self._entities_results = None
        self.intent_results = intent_result
        self.entities_results = entity_results

    @property
    def intent_result(self):
        return self._intent_result

    @intent_result.setter
    def intent_result(self, value):
        if not isinstance(value, IntentClassificationResult):
            raise ValueError("Expected IntentClassificationResult, found: %s"
                             % type(value))

    @property
    def entities(self):
        return self._entities_results

    @entities.setter
    def entities(self, values):
        if not isinstance(values, EntityExtractionResult):
            raise ValueError("entities must be a EntityExtractionResult"
                             " instances")
        self._entities_results = values

    def to_dict(self):
        result = {
            "text": self.text,
            "entities": self.entities_results.as_list(),
            "intent": {}
        }
        intent_as_dict = self.intent_result.to_dict()
        if "intent" in intent_as_dict:
            result["intent"]["name"] = intent_as_dict["intent"]
            prob = intent_as_dict.get("prob", False)
            if prob:
                result["intent"]["prob"] = prob
        return result
