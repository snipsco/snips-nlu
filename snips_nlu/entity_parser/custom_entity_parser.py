# coding=utf-8
from __future__ import unicode_literals

import json
from copy import deepcopy
from pathlib import Path

from future.utils import iteritems, viewvalues
from snips_nlu_ontology import GazetteerEntityParser

from snips_nlu.constants import (
    END, ENTITIES, LANGUAGE, PARSER_THRESHOLD, RES_MATCH_RANGE, START,
    UTTERANCES, ENTITY_KIND)
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.entity_parser.custom_entity_parser_usage import (
    CustomEntityParserUsage)
from snips_nlu.entity_parser.entity_parser import EntityParser
from snips_nlu.preprocessing import stem, tokenize
from snips_nlu.utils import json_string


class CustomEntityParser(EntityParser):
    def __init__(self, parser, language, parser_usage):
        super(CustomEntityParser, self).__init__(parser)
        self.language = language
        self.parser_usage = parser_usage

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        parser_directory = "parser"
        metadata = {
            "language": self.language,
            "parser_usage": self.parser_usage.value,
            "parser_directory": parser_directory
        }
        with (path / "metadata.json").open(mode="w", encoding="utf8") as f:
            f.write(json_string(metadata))
        self._parser.persist(path / parser_directory)

    @classmethod
    def from_path(cls, path):
        path = Path(path)
        with (path / "metadata.json").open(encoding="utf8") as f:
            metadata = json.load(f)
        language = metadata["language"]
        parser_usage = CustomEntityParserUsage(metadata["parser_usage"])
        parser_path = path / metadata["parser_directory"]
        parser = GazetteerEntityParser.from_path(parser_path)
        return cls(parser, language, parser_usage)

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
                stemmed_utterances = _stem_entity_utterances(
                    ent[UTTERANCES], language)
                ent[UTTERANCES] = _merge_entity_utterances(
                    ent[UTTERANCES], stemmed_utterances)
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
        return cls(parser, language, parser_usage)

    def parse(self, text, scope=None, use_cache=True):
        text = text.lower()
        if not use_cache:
            return self._parse(text, scope)
        scope_key = tuple(sorted(scope)) if scope is not None else scope
        cache_key = (text, scope_key)
        if cache_key not in self._cache:
            parser_result = self._parse(text, scope)
            self._cache[cache_key] = parser_result
        return deepcopy(self._cache[cache_key])

    def _parse(self, text, scope):
        tokens = tokenize(text, self.language)
        shifts = _compute_char_shifts(tokens)
        cleaned_text = " ".join(token.value for token in tokens)
        entities = self._parser.parse(cleaned_text, scope)
        for entity in entities:
            start = entity[RES_MATCH_RANGE][START]
            end = entity[RES_MATCH_RANGE][END]
            entity[ENTITY_KIND] = entity.pop("entity_identifier")
            entity[RES_MATCH_RANGE][START] -= shifts[start]
            entity[RES_MATCH_RANGE][END] -= shifts[end - 1]
        return entities


def _stem_entity_utterances(entity_utterances, language):
    return {
        stem(raw_value, language): resolved_value
        for raw_value, resolved_value in iteritems(entity_utterances)
    }


def _merge_entity_utterances(raw_utterances, stemmed_utterances):
    for raw_stemmed_value, resolved_value in iteritems(stemmed_utterances):
        if raw_stemmed_value not in raw_utterances:
            raw_utterances[raw_stemmed_value] = resolved_value
    return raw_utterances


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


def _compute_char_shifts(tokens):
    """Compute the shifts in characters that occur when comparing the
    tokens string with the string consisting of all tokens separated with a
    space

    For instance, if "hello?world" is tokenized in ["hello", "?", "world"],
    then the character shifts between "hello?world" and "hello ? world" are
    [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2]
    """
    characters_shifts = []
    if not tokens:
        return characters_shifts

    current_shift = 0
    for token_index, token in enumerate(tokens):
        if token_index == 0:
            previous_token_end = 0
            previous_space_len = 0
        else:
            previous_token_end = tokens[token_index - 1].end
            previous_space_len = 1
        current_shift -= (token.start - previous_token_end) - \
                         previous_space_len
        token_len = token.end - token.start
        index_shift = token_len + previous_space_len
        characters_shifts += [current_shift for _ in range(index_shift)]
    return characters_shifts
