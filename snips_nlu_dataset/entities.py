# coding=utf-8
from __future__ import unicode_literals

import csv
import io
import os
from abc import ABCMeta, abstractmethod

import six
from future.utils import with_metaclass

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (
    VALUE, SYNONYMS, AUTOMATICALLY_EXTENSIBLE, USE_SYNONYMS, DATA)


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
                 use_synonyms):
        super(CustomEntity, self).__init__(name)
        self.utterances = utterances
        self.automatically_extensible = automatically_extensible
        self.use_synonyms = use_synonyms

    @classmethod
    def from_file(cls, entity_file_name):
        entity_name = ".".join(
            os.path.basename(entity_file_name).split('.')[:-1])
        utterances = []
        with io.open(entity_file_name, "r", encoding="utf-8") as f:
            it = f
            if six.PY2:
                it = list(utf_8_encoder(it))
            reader = csv.reader(list(it))
            for row in reader:
                if six.PY2:
                    row = [cell.decode("utf-8") for cell in row]
                value = row[0]
                if len(row) > 1:
                    synonyms = row[1:]
                else:
                    synonyms = []
                utterances.append(EntityUtterance(value, synonyms))
        return cls(entity_name, utterances, automatically_extensible=True,
                   use_synonyms=True)

    @property
    def json(self):
        """Returns the entity in json format"""
        return {
            AUTOMATICALLY_EXTENSIBLE: self.automatically_extensible,
            USE_SYNONYMS: self.use_synonyms,
            DATA: [u.json for u in self.utterances]
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
