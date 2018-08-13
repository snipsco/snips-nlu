# coding=utf-8
from __future__ import unicode_literals

import json
from pathlib import Path

from enum import Enum, unique
from future.utils import iteritems, viewvalues
from snips_nlu_ontology import GazetteerEntityParser

from snips_nlu.constants import (
    ENTITIES, LANGUAGE, PARSER_THRESHOLD, STEMS, UTTERANCES)
from snips_nlu.parser.builtin_entity_parser import is_gazetteer_entity
from snips_nlu.parser.entity_parser import (
    EntityParser)
from snips_nlu.pipeline.configs import ProcessingUnitConfig
from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.preprocessing import stem
from snips_nlu.utils import classproperty, json_string


@unique
class EntityStemsUsage(Enum):
    STEMS = 0
    """Usage of stemming"""
    NO_STEMS = 1
    """No usage of stemming"""
    STEMS_AND_NO_STEMS = 2
    """Usage of both stemming and not stemming"""


class CustomEntityParserConfig(ProcessingUnitConfig):

    def __init__(self, stemming_usage):
        self.stemming_usage = stemming_usage

    def to_dict(self):
        return {"stemming_usage": self.stemming_usage.value}

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(stemming_usage=obj_dict["stemming_usage"])

    def get_required_resources(self):
        return {STEMS: self.stemming_usage}

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        return CustomEntityParser.unit_name


class CustomEntityParser(ProcessingUnit, EntityParser):
    config_type = CustomEntityParserConfig
    unit_name = "custom_entity_parser"

    # pylint: disable=super-init-not-called
    def __init__(self, config, **shared):
        self.config = config
        self._parser = None
        self.entities = None

    @property
    def parser(self):
        return self._parser

    @property
    def fitted(self):
        return self._parser is not None

    def parse(self, text, scope=None, use_cache=True):
        if self.parser is None:
            raise RuntimeError("Custom entity parser must be fitted on a"
                               " dataset before parsing")
        return super(CustomEntityParser, self).parse(
            text, scope=scope, use_cache=use_cache)

    def fit(self, dataset):
        language = dataset[LANGUAGE]
        entities = {
            entity_name: entity
            for entity_name, entity in iteritems(dataset[ENTITIES])
            if not is_gazetteer_entity(entity_name)
        }
        self.entities = [name for name in entities]
        if self.config.stemming_usage == EntityStemsUsage.STEMS_AND_NO_STEMS:
            for ent in viewvalues(entities):
                ent[UTTERANCES].update(
                    _stem_entity_utterances(ent[UTTERANCES], language))
        elif self.config.stemming_usage == EntityStemsUsage.STEMS:
            for ent in viewvalues(entities):
                ent[UTTERANCES] = _stem_entity_utterances(
                    ent[UTTERANCES], language)
        configurations = _create_custom_entity_parser_configurations(entities)
        self._parser = GazetteerEntityParser(configurations)
        return self

    def persist(self, path):
        path = Path(path)
        metadata = {"entities": self.entities}
        metadata_string = json_string(metadata)
        metadata_path = path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(metadata_string)
        if self.parser is not None:
            parser_path = path / "parser"
            self.parser.dump(str(parser_path))

    # pylint: disable=protected-access
    @classmethod
    def from_path(cls, path, **shared):
        parser_path = Path(path) / "parser"
        self = cls(None)
        self._parser = None
        if parser_path.exists():
            self._parser = GazetteerEntityParser.load(parser_path)
        metadata_path = path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            metadata = json.load(f)
        self.entities = metadata["entities"]
        return self


def get_custom_entity_parser(dataset):
    return CustomEntityParser(None).fit(dataset)


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
