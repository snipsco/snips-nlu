# coding=utf-8
from __future__ import unicode_literals

import csv
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

import six
from future.utils import with_metaclass

from snips_nlu.constants import (AUTOMATICALLY_EXTENSIBLE, DATA,
                                 PARSER_THRESHOLD, SYNONYMS, USE_SYNONYMS,
                                 VALUE)
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity

AUTO_EXT_REGEX = re.compile(r'^#\sautomatically_extensible=(true|false)\s*$')


class Entity(with_metaclass(ABCMeta, object)):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def json(self):
        pass


class CustomEntity(Entity):
    """Custom entity of an :class:`.AssistantDataset`

        Attributes:
            utterances (list of :class:`.EntityUtterance`): entity utterances
            automatically_extensible (bool): whether or not the entity can be
                extended to values not present in the dataset
            use_synonyms (bool): whether or not to map entity values using
                synonyms
    """

    def __init__(self, name, utterances, automatically_extensible,
                 use_synonyms, parser_threshold=1.0):
        super(CustomEntity, self).__init__(name)
        self.utterances = utterances
        self.automatically_extensible = automatically_extensible
        self.use_synonyms = use_synonyms
        self.parser_threshold = parser_threshold

    @classmethod
    def from_file(cls, filepath):
        filepath = Path(filepath)
        stem = filepath.stem
        if not stem.startswith("entity_"):
            raise AssertionError("Entity filename should start with 'entity_' "
                                 "but found: %s" % stem)
        entity_name = stem[7:]
        if not entity_name:
            raise AssertionError("Entity name must not be empty")
        utterances = []
        with filepath.open(encoding="utf-8") as f:
            it = f
            if six.PY2:
                it = list(utf_8_encoder(it))
            reader = csv.reader(list(it))
            autoextent = True
            for row in reader:
                if six.PY2:
                    row = [cell.decode("utf-8") for cell in row]
                value = row[0]
                if reader.line_num == 1:
                    m = AUTO_EXT_REGEX.match(row[0])
                    if m:
                        autoextent = not m.group(1).lower() == 'false'
                        continue
                if len(row) > 1:
                    synonyms = row[1:]
                else:
                    synonyms = []
                utterances.append(EntityUtterance(value, synonyms))
        return cls(entity_name, utterances,
                   automatically_extensible=autoextent, use_synonyms=True)

    @property
    def json(self):
        """Returns the entity in json format"""
        return {
            AUTOMATICALLY_EXTENSIBLE: self.automatically_extensible,
            USE_SYNONYMS: self.use_synonyms,
            DATA: [u.json for u in self.utterances],
            PARSER_THRESHOLD: self.parser_threshold
        }


class EntityUtterance(object):
    """Represents a value of a :class:`.CustomEntity` with potential synonyms

    Attributes:
        value (str): entity value
        synonyms (list of str): The values to remap to the utterance value
        """

    def __init__(self, value, synonyms=None):
        self.value = value
        if synonyms is None:
            synonyms = []
        self.synonyms = synonyms

    @property
    def json(self):
        return {VALUE: self.value, SYNONYMS: self.synonyms}


class BuiltinEntity(Entity):
    """Builtin entity of an :class:`.AssistantDataset`"""

    @property
    def json(self):
        return dict()


def utf_8_encoder(f):
    for line in f:
        yield line.encode("utf-8")


def create_entity(entity_name, utterances=None, automatically_extensible=True,
                  use_synonyms=True):
    if is_builtin_entity(entity_name):
        return BuiltinEntity(entity_name)
    else:
        if utterances is None:
            utterances = []
        return CustomEntity(entity_name, utterances, automatically_extensible,
                            use_synonyms)
