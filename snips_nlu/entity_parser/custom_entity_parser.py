# coding=utf-8
from __future__ import unicode_literals

import json
from enum import Enum, unique
from pathlib import Path

from future.builtins import str
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
        self.entities = set(entities)
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
        path = Path(path)
        _parser_path = None
        if self.parser is not None:
            _parser_path = str(path / "parser")
            self.parser.dump(str(_parser_path))
        parser_model = {
            "entities": list(self.entities),
            "parser": _parser_path,
            "parser_usage": self.parser_usage,
        }
        parser_path = path / "custom_entity_parser.json"
        with parser_path.open("w", encoding="utf-8") as f:
            f.write(json_string(parser_model))
        self.persist_metadata(path)

    # pylint: disable=protected-access
    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)

        model_path = path / "custom_entity_parser.json"
        with model_path.open("r", encoding="utf-8") as f:
            model = json.load(f)

        parser_usage = CustomEntityParserUsage(model["parser_usage"])
        custom_parser = CustomEntityParser(parser_usage)
        custom_parser.entities = set(model["entities"])

        _parser_path = Path(model["parser"])
        if _parser_path.exists():
            custom_parser._parser = GazetteerEntityParser.load(_parser_path)

        return custom_parser


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
