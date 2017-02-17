import io
import json
import os
import re
from copy import deepcopy

from entity import Entity

VALID_FILE_REGEX = re.compile(r"^[\w\s-]+$")


def is_valid_filename(string):
    return VALID_FILE_REGEX.match(string)


def format_query_for_import(query):
    formatted_query = deepcopy(query)
    for chunk in formatted_query["data"]:
        if "entity" in chunk:
            chunk["entity"] = chunk["entity"].lstrip("@")
    return formatted_query


def format_query_for_export(query):
    formatted_query = deepcopy(query)
    for chunk in formatted_query["data"]:
        if "entity" in chunk:
            chunk["entity"] = "@" + chunk["entity"]
    return formatted_query


def validate_queries(queries, entities):
    entities_names = entities.keys()
    for intent_name, intent_queries in queries.iteritems():
        # TODO: find a better way to ensure intent_name validity
        if not is_valid_filename(intent_name):
            raise ValueError("%s is an invalid intent name. Intent names must "
                             "be a valid file name, use only: [a-zA-Z0-9_- ]"
                             % intent_name)
        for q in intent_queries:
            data = q["data"]
            for chunk in data:
                if "text" not in chunk:
                    raise ValueError("Query chunk must have 'text' key")

                entity_name = chunk.get("entity", None)
                if entity_name is not None:
                    if entity_name not in entities_names:
                        raise ValueError("Unknown entity '%s'. Entities must "
                                         "belong to %s" %
                                         (entity_name, entities_names))


class Dataset(object):
    def __init__(self, entities=None, queries=None):
        self._entities = {}
        self._queries = {}
        if entities is not None:
            self.entities = entities
        if queries is not None:
            self.queries = queries

    def __eq__(self, other):
        if self.queries != other.queries:
            return False
        if self.entities != self.entities:
            return False
        return True

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, value):
        if isinstance(value, dict):
            for ent in value.values():
                assert isinstance(ent, Entity)
            self._entities = value
        elif isinstance(value, list):
            for ent in value:
                assert isinstance(ent, Entity)
            self._entities = dict((ent.name, ent) for ent in value)
        else:
            raise ValueError("Expected dict or list, found %s" % type(value))

    @property
    def queries(self):
        return self._queries

    @queries.setter
    def queries(self, value):
        validate_queries(value, self.entities)
        self._queries = value

    @classmethod
    def load(cls, dir_path):
        queries_path = os.path.join(dir_path, "queries")
        json_files = [f for f in os.listdir(queries_path)
                      if f.endswith("json")]
        queries = dict()
        for f in json_files:
            intent_name, _ = os.path.splitext(f)
            intent_queries = cls.load_intent_queries(
                os.path.join(queries_path, f))
            intent_queries = [format_query_for_import(q)
                              for q in intent_queries]
            queries[intent_name] = intent_queries

        entities_path = os.path.join(dir_path, "entities")
        json_files = [f for f in os.listdir(entities_path)
                      if f.endswith("json")]
        entities = dict()
        for f in json_files:
            entity = Entity.from_json(os.path.join(entities_path, f))
            entities[entity.name] = entity

        return cls(entities=entities, queries=queries)

    @staticmethod
    def load_intent_queries(path):
        expected_intent_name, _ = os.path.splitext(os.path.basename(path))
        with io.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["intent"] != expected_intent_name:
            raise ValueError("query file name and intent name must match. "
                             "Expected intent %s, found %s"
                             % (expected_intent_name, data["intent"]))

        return data["queries"]

    def save(self, dir_path):
        if os.path.exists(dir_path):
            raise RuntimeError("%s is an existing directory or file"
                               % dir_path)
        os.mkdir(dir_path)

        queries_path = os.path.join(dir_path, "queries")
        os.mkdir(queries_path)
        for intent_name, intent_queries in self.queries.iteritems():
            path = os.path.join(queries_path, "%s.json" % intent_name)
            data = {
                "intent": intent_name,
                "queries": [format_query_for_export(q)
                            for q in intent_queries],
            }
            with io.open(path, "w", encoding="utf-8") as f:
                data = json.dumps(data, indent=2)
                f.write(unicode(data))

        entities_path = os.path.join(dir_path, "entities")
        os.mkdir(entities_path)
        for entity_name, entity in self.entities.iteritems():
            path = os.path.join(entities_path,
                                "%s.json" % entity_name)
            entity.to_json(path)
