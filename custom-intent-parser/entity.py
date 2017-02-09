import json


class Entity(object):
    def __init__(self, entries=None, is_extensible=False):
        self.entries = entries if entries is not None else []
        self.is_extensible = is_extensible

    def to_dict(self):
        return {
            "is_extensible": self.is_extensible,
            "entries": self.entries
        }

    @classmethod
    def from_dict(cls, dict_entity):
        dict_keys = {"is_extensible", "entries"}
        if not set(dict_entity.keys()) == dict_keys:
            raise ValueError("Entity dict keys must exactly be %s" % dict_keys)

        is_extensible = dict_entity["is_extensible"]
        if not type(is_extensible) == bool:
            raise ValueError("is_extensible must be a bool, found %s"
                             % type(is_extensible))

        entries = dict_entity["entries"]
        if not type(entries) == list:
            raise ValueError("entries must be a list, found %s"
                             % type(entries))

        for e in entries:
            cls.validate_entry(e)

        return cls(entries=entries, is_extensible=is_extensible)

    @classmethod
    def from_json(cls, json_string):
        return cls.from_dict(json.loads(json_string))

    @staticmethod
    def validate_entry(dict_entry):
        if not dict_entry.keys() == ["value"]:
            raise ValueError("Entities entry should be a dict with a single "
                             "key: 'value'")
