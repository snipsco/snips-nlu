# coding=utf-8
from __future__ import unicode_literals

from copy import deepcopy

from future.utils import iteritems, viewvalues
from snips_nlu_ontology import GazetteerEntityParser

from snips_nlu.constants import (
    ENTITIES, LANGUAGE, PARSER_THRESHOLD, UTTERANCES)
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.entity_parser.custom_entity_parser_usage import (
    CustomEntityParserUsage)
from snips_nlu.entity_parser.entity_parser import EntityParser
from snips_nlu.preprocessing import stem


class CustomEntityParser(EntityParser):
    @classmethod
    def from_path(cls, path):
        parser = GazetteerEntityParser.from_path(path)
        return cls(parser)

    @classmethod
    def build(cls, dataset, parser_usage):
        from snips_nlu.dataset import validate_and_format_dataset

        dataset = validate_and_format_dataset(dataset)
        language = dataset[LANGUAGE]
        custom_entities = {
            entity_name: deepcopy(entity)
            for entity_name, entity in iteritems(dataset[ENTITIES])
            if not is_builtin_entity(entity_name)
        }
        if parser_usage == CustomEntityParserUsage.WITH_AND_WITHOUT_STEMS:
            for ent in viewvalues(custom_entities):
                ent[UTTERANCES].update(
                    _stem_entity_utterances(ent[UTTERANCES], language))
        elif parser_usage == CustomEntityParserUsage.WITH_STEMS:
            for ent in viewvalues(custom_entities):
                ent[UTTERANCES] = _stem_entity_utterances(
                    ent[UTTERANCES], language)
        elif parser_usage is None:
            raise ValueError("A parser usage must be defined in order to fit "
                             "a CustomEntityParser")
        configuration = _create_custom_entity_parser_configuration(
            custom_entities)
        parser = GazetteerEntityParser.build(configuration)
        return cls(parser)


def _stem_entity_utterances(entity_utterances, language):
    return {
        stem(raw_value, language): resolved_value
        for raw_value, resolved_value in iteritems(entity_utterances)
    }


def _create_custom_entity_parser_configuration(entities):
    return {
        "entity_parsers": [
            {
                "entity_identifier": entity_name,
                "entity_parser": {
                    "threshold": entity[PARSER_THRESHOLD],
                    "gazetteer": [
                        {
                            "raw_value": k,
                            "resolved_value": v
                        } for k, v in iteritems(entity[UTTERANCES])
                    ]
                }
            } for entity_name, entity in iteritems(entities)
        ]
    }
