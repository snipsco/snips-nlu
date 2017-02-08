import io
import json
import os


def is_valid_filename(string):
    return "\\" not in string and "/" not in string


def validate_queries(queries):
    for intent_name in queries:
        if not is_valid_filename(intent_name):
            raise ValueError("%s is an invalid intent name. Intent names must "
                             "be a valid file name: no slash or backslash.")


class Dataset(object):
    def __init__(self, queries):
        # TODO: @ClemDoum maybe not the best place to ensure consistency
        validate_queries(queries)
        self.queries = queries

    @classmethod
    def load(cls, dir_path):
        json_files = [f for f in os.listdir(dir_path) if f.endswith("json")]
        queries = [(f, cls.load_intent_queries(os.path.join(dir_path, f)))
                   for f in json_files]
        return cls(queries)

    @staticmethod
    def load_intent_queries(path):
        with io.open(path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        return queries

    def save(self, dir_path):
        if os.path.exists(dir_path):
            raise RuntimeError("%s is an existing directory or file"
                               % dir_path)
        os.mkdir(dir_path)
        for intent, intent_queries in self.queries.iteritems():
            queries_path = os.path.join(dir_path, "%s.json" % intent)
            with io.open(queries_path, "w", encoding="utf-8") as f:
                json.dump(self.queries[intent], f, intent=2)
