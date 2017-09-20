from __future__ import unicode_literals

import string

from enum import Enum
from zhon import hanzi

from utils import classproperty, regex_escape

ISO_CODE, RUSTLING_CODE = "iso", "rustling_code"

SPACE = " "
WHITE_SPACES = "%s\t\n\r\f\v" % SPACE  # equivalent of r"\s"
COMMONLY_IGNORED_CHARACTERS = "%s%s" % (WHITE_SPACES, string.punctuation)
COMMONLY_IGNORED_CHARACTERS_PATTERN = r"[%s]*" % regex_escape(
    COMMONLY_IGNORED_CHARACTERS)

MANDARIN_PUNCTUATION = hanzi.punctuation + string.punctuation
MANDARIN_IGNORED_CHARACTERS = "%s%s" % (WHITE_SPACES, MANDARIN_PUNCTUATION)
MANDARIN_IGNORED_CHARACTERS_PATTERN = r"[%s]*" % regex_escape(
    MANDARIN_IGNORED_CHARACTERS)


class Language(Enum):
    EN = {ISO_CODE: "en", RUSTLING_CODE: "EN"}
    ES = {ISO_CODE: "es", RUSTLING_CODE: "ES"}
    FR = {ISO_CODE: "fr", RUSTLING_CODE: "FR"}
    DE = {ISO_CODE: "de", RUSTLING_CODE: "DE"}
    KO = {ISO_CODE: "ko", RUSTLING_CODE: "KO"}
    ZH = {ISO_CODE: "zh", RUSTLING_CODE: "ZH"}

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

    @property
    def default_sep(self):
        if self == Language.ZH:
            return ""
        else:
            return " "

    @property
    def punctuation(self):
        if self == Language.ZH:
            return MANDARIN_PUNCTUATION
        else:
            return string.punctuation

    @property
    def ignored_characters_pattern(self):
        if self == Language.ZH:
            return MANDARIN_IGNORED_CHARACTERS_PATTERN
        else:
            return COMMONLY_IGNORED_CHARACTERS_PATTERN


NON_SEGMENTED_LANGUAGES = {Language.ZH}
