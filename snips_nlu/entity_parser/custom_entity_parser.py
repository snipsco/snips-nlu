# coding=utf-8
from __future__ import unicode_literals

import json
from enum import Enum, unique
from pathlib import Path

from future.utils import iteritems, viewvalues
from snips_nlu_ontology import GazetteerEntityParser

from snips_nlu.constants import (
    ENTITIES, LANGUAGE, PARSER_THRESHOLD, UTTERANCES, CUSTOM_ENTITY_PARSER)
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.entity_parser.entity_parser import (
    EntityParser)
from snips_nlu.pipeline.processing_unit import SerializableUnit
from snips_nlu.preprocessing import stem
from snips_nlu.utils import NotTrained, json_string


@unique
class CustomEntityParserUsage(Enum):
    WITH_STEMS = 0
    """The parser is used with stemming"""
    WITHOUT_STEMS = 1
    """The parser is used without stemming"""
    WITH_AND_WITHOUT_STEMS = 2
    """The parser is used both with and without stemming"""

    @classmethod
    def merge_usages(cls, lhs_usage, rhs_usage):
        if lhs_usage is None:
            return rhs_usage
        if rhs_usage is None:
            return lhs_usage
        if lhs_usage == rhs_usage:
            return lhs_usage
        return cls.WITH_AND_WITHOUT_STEMS


class CustomEntityParser(EntityParser, SerializableUnit):
    unit_name = CUSTOM_ENTITY_PARSER

    def __init__(self, parser_usage):
        self.parser_usage = parser_usage
        self._parser = None
        self.entities = None

    @property
    def parser(self):
        return self._parser

    @property
    def fitted(self):
        return self._parser is not None

    def parse(self, text, scope=None, use_cache=True):
        if not self.fitted:
            raise NotTrained("CustomEntityParser must be fitted")
        return super(CustomEntityParser, self).parse(
            text, scope=scope, use_cache=use_cache)

    def fit(self, dataset):
        language = dataset[LANGUAGE]
        entities = {
            entity_name: entity
            for entity_name, entity in iteritems(dataset[ENTITIES])
            if not is_builtin_entity(entity_name)
        }
        self.entities = [name for name in entities]
        if self.parser_usage == CustomEntityParserUsage.WITH_AND_WITHOUT_STEMS:
            for ent in viewvalues(entities):
                ent[UTTERANCES].update(
                    _stem_entity_utterances(ent[UTTERANCES], language))
        elif self.parser_usage == CustomEntityParserUsage.WITH_STEMS:
            for ent in viewvalues(entities):
                ent[UTTERANCES] = _stem_entity_utterances(
                    ent[UTTERANCES], language)
        elif self.parser_usage is None:
            raise ValueError("A parser usage must be defined in order to fit "
                             "a CustomEntityParser")
        configurations = _create_custom_entity_parser_configurations(entities)
        self._parser = GazetteerEntityParser(configurations)
        return self

    def persist(self, path):
        if self.parser is not None:
            parser_path = path / "parser"
            self.parser.dump(str(parser_path))
        self.persist_metadata(path)

    def persist_metadata(self, path, **kwargs):
        path = Path(path)
        metadata = {"entities": self.entities, "unit_name": self.unit_name}
        metadata_string = json_string(metadata)
        metadata_path = path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(metadata_string)

    # pylint: disable=protected-access
    @classmethod
    def from_path(cls, path, **shared):
        parser_path = Path(path) / "parser"
        custom_parser = cls(None)
        custom_parser._parser = None
        if parser_path.exists():
            custom_parser._parser = GazetteerEntityParser.load(parser_path)
        metadata_path = path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            metadata = json.load(f)
        custom_parser.entities = metadata["entities"]
        return custom_parser


def get_custom_entity_parser(dataset, parser_usage):
    return CustomEntityParser(parser_usage).fit(dataset)


def _stem_entity_utterances(entity_utterances, language):
    return {
        stem(raw_value, language): resolved_value
        for raw_value, resolved_value in iteritems(entity_utterances)
    }


def _create_custom_entity_parser_configurations(entities):
    return {
        entity: {
            "parser_threshold": entity[PARSER_THRESHOLD],
            "entity_values": entity[UTTERANCES]
        } for entity in entities
    }
