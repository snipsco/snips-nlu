import io
import json
import re

from utils import sequence_equal

ALPHA_REGEX = re.compile("^[\w]+$")


def check_type(obj, allowed_types, obj_label=None):
    if obj_label is None:
        obj_label = "object"
    if not type(obj) in allowed_types:
        raise ValueError(
            "Expected %s to be a %s, but found %s" %
            (obj_label, [a.__name__ for a in allowed_types],
             type(obj).__name__))


class Entity(object):
    def __init__(self, name, entries=None, use_learning=False,
                 use_synonyms=False):
        self.name = name
        entries = entries if entries is not None else []
        self.validate_entries(entries)
        self.entries = entries
        self.use_learning = use_learning
        self.use_synonyms = use_synonyms

    def __eq__(self, other):
        if self._name != other.name:
            return False
        if self.use_learning != other.use_learning:
            return False
        if self.use_synonyms != other.use_synonyms:
            return False
        if not sequence_equal(self.entries, other.entries):
            return False
        return True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not ALPHA_REGEX.match(value):
            raise ValueError("Entity name must only contain [0-9a-zA-Z_],"
                             " found '%s'" % value)
        self._name = value

    def to_dict(self):
        return {
            "name": self.name,
            "automaticallyExtensible": self.use_learning,
            "useSynonyms": self.use_synonyms,
            "entries": self.entries
        }

    def to_json(self, path):
        with io.open(path, "w", encoding="utf-8") as f:
            data = json.dumps(self.to_dict(), indent=2)
            f.write(unicode(data))

    @classmethod
    def from_dict(cls, dict_entity):
        dict_keys = {"name", "automaticallyExtensible", "useSynonyms",
                     "entries"}
        if not set(dict_entity.keys()) == dict_keys:
            raise ValueError("Entity dict keys must exactly be %s" % dict_keys)

        name = dict_entity["name"]
        check_type(name, [str, unicode], obj_label="name")

        use_learning = dict_entity["automaticallyExtensible"]
        check_type(use_learning, [bool], obj_label="automaticallyExtensible")

        use_synonyms = dict_entity["useSynonyms"]
        check_type(use_synonyms, [bool], obj_label="useSynonyms")

        entries = dict_entity["entries"]
        check_type(entries, [list], obj_label=entries)

        return cls(name, entries=entries, use_learning=use_learning,
                   use_synonyms=use_synonyms)

    @classmethod
    def from_json(cls, path):
        with io.open(path, encoding="utf-8") as f:
            entity_as_dict = json.load(f)
        return cls.from_dict(entity_as_dict)

    @staticmethod
    def validate_entries(entries):
        for entry in entries:
            expected_keys = {"value", "synonyms"}
            actual_keys = set(entry.keys())

            unexpected_keys = actual_keys - expected_keys
            if len(unexpected_keys) > 0:
                raise ValueError("Unexpected entry keys: %s" % unexpected_keys)

            missing_keys = expected_keys - actual_keys
            if len(missing_keys) > 0:
                raise ValueError("Missing entry keys: %s" % str(missing_keys))

            value = entry["value"]
            synonyms = entry["synonyms"]
            if len(synonyms) == 1:
                if not synonyms[0] == value:
                    raise ValueError("If there only one synonym it must be"
                                     " equal to the 'value' field")
            elif len(synonyms) == 0:
                raise ValueError("There must be at least one synonym equal to "
                                 "'value' field")
