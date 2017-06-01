from __future__ import unicode_literals

from enum import Enum

from utils import classproperty

ISO_CODE, RUSTLING_CODE = "iso", "rustling_code"


class Language(Enum):
    EN = {ISO_CODE: "en", RUSTLING_CODE: "EN"}
    ES = {ISO_CODE: "es", RUSTLING_CODE: "ES"}
    FR = {ISO_CODE: "fr", RUSTLING_CODE: "FR"}
    DE = {ISO_CODE: "de", RUSTLING_CODE: "DE"}
    KO = {ISO_CODE: "ko", RUSTLING_CODE: "KO"}

    @property
    def iso_code(self):
        return self.value[ISO_CODE]

    @property
    def rustling_code(self):
        return self.value[RUSTLING_CODE]

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
                raise KeyError("Unknown iso_code '%s'" % iso_code)
            else:
                return default
        return ent

    @classproperty
    @classmethod
    def language_by_rustling_code(cls):
        try:
            return cls._language_by_rustling_code
        except AttributeError:
            cls._language_by_rustling_code = dict()
            for ent in cls:
                cls._language_by_rustling_code[ent.rustling_code] = ent
        return cls._language_by_rustling_code

    @classmethod
    def from_rustling_code(cls, rustling_code, default=None):
        try:
            ent = cls.language_by_rustling_code[rustling_code]
        except KeyError:
            if default is None:
                raise KeyError("Unknown rustling_code '%s'" % rustling_code)
            else:
                return default
        return ent
