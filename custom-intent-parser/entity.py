import json


class Entity(object):
    def __init__(self, entries=None, is_keyword=False, is_enum=False):
        self.entries = entries if entries is not None else []
        self.is_keyword = is_keyword
        self.is_enum = is_enum

    def to_dict(self):
        return {
            "isKeyword": self.is_keyword,
            "isEnum": self.is_enum,
            "entries": self.entries
        }

    @classmethod
    def from_dict(cls, dict_entity):
        dict_keys = {"isKeyword", "isEnum", "entries"}
        if not set(dict_entity.keys()) == dict_keys:
            raise ValueError("Entity dict keys must exactly be %s" % dict_keys)

        is_keyword = dict_entity["isKeyword"]
        if not type(is_keyword) == bool:
            raise ValueError("isKeyword must be a bool, found %s"
                             % type(is_keyword))

        is_enum = dict_entity["isEnum"]
        if not type(is_enum) == bool:
            raise ValueError("isEnum must be a bool, found %s" % type(is_enum))

        entries = dict_entity["entries"]
        if not type(entries) == list:
            raise ValueError("entries must be a list, found %s"
                             % type(entries))

        for e in entries:
            cls.validate_entry(e)

        return cls(entries=entries, is_keyword=is_keyword, is_enum=is_enum)

    @classmethod
    def from_json(cls, json_string):
        return cls.from_dict(json.loads(json_string))

    @staticmethod
    def validate_entry(dict_entry):
        if not dict_entry.keys() == ["value"]:
            raise ValueError("Entities entry should be a dict with a single "
                             "key: 'value'")
