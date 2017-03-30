from duckling import core
from enum import Enum

from utils import classproperty

core.load()

ISO_CODE, DUCKLING_CODE = "iso", "ducking_code"


class Language(Enum):
    ENG = {ISO_CODE: "eng", DUCKLING_CODE: "en"}
    SPA = {ISO_CODE: "spa", DUCKLING_CODE: "es"}
    FRA = {ISO_CODE: "fra", DUCKLING_CODE: "fr"}

    @property
    def iso_code(self):
        return self.value[ISO_CODE]

    @property
    def duckling_code(self):
        return self.value[DUCKLING_CODE]

    @classproperty
    @classmethod
    def language_by_iso_code(cls):
        try:
            return cls._language_by_iso_code
        except AttributeError:
            cls._language_by_iso_code = dict()
            for ent in cls:
                cls._language_by_iso_code[ent.iso_code] = ent
        return cls._language_by_iso_code

    @classmethod
    def from_iso_code(cls, iso_code, default=None):
        try:
            ent = cls.language_by_iso_code[iso_code]
        except KeyError:
            if default is None:
                raise KeyError("Unknown entity '%s'" % iso_code)
            else:
                return default
        return ent

    @classproperty
    @classmethod
    def language_by_duckling_code(cls):
        try:
            return cls._language_by_duckling_code
        except AttributeError:
            cls._language_by_duckling_code = dict()
            for ent in cls:
                cls._language_by_duckling_code[ent.duckling_code] = ent
        return cls._language_by_duckling_code

    @classmethod
    def from_duckling_code(cls, duckling_code, default=None):
        try:
            ent = cls.language_by_duckling_code[duckling_code]
        except KeyError:
            if default is None:
                raise KeyError("Unknown duckling '%s'" % duckling_code)
            else:
                return default
        return ent
