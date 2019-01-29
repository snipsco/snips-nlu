# coding=utf-8
from __future__ import unicode_literals

import json
import operator
from copy import deepcopy
from pathlib import Path

from future.utils import iteritems, viewvalues
from snips_nlu_parsers import GazetteerEntityParser

from snips_nlu.common.utils import json_string
from snips_nlu.constants import (
    END, ENTITIES, LANGUAGE, MATCHING_STRICTNESS, START, UTTERANCES)
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.entity_parser.custom_entity_parser_usage import (
    CustomEntityParserUsage)
from snips_nlu.entity_parser.entity_parser import EntityParser
from snips_nlu.preprocessing import stem, tokenize
from snips_nlu.result import parsed_entity


class CustomEntityParser(EntityParser):
    def __init__(self, parser, language, parser_usage):
        super(CustomEntityParser, self).__init__()
        self._parser = parser
        self.language = language
        self.parser_usage = parser_usage

    def _parse(self, text, scope=None):
        tokens = tokenize(text, self.language)
        shifts = _compute_char_shifts(tokens)
        cleaned_text = " ".join(token.value for token in tokens)

        entities = self._parser.parse(cleaned_text, scope)
        result = []
        for entity in entities:
            start = entity["range"]["start"]
            start -= shifts[start]
            end = entity["range"]["end"]
            end -= shifts[end - 1]
            entity_range = {START: start, END: end}
            ent = parsed_entity(
                entity_kind=entity["entity_identifier"],
                entity_value=entity["value"],
                entity_resolved_value=entity["resolved_value"],
                entity_range=entity_range
            )
            result.append(ent)
        return result

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
    def build(cls, dataset, parser_usage, resources):
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
                    ent[UTTERANCES], language, resources)
                ent[UTTERANCES] = _merge_entity_utterances(
                    ent[UTTERANCES], stemmed_utterances)
        elif parser_usage == CustomEntityParserUsage.WITH_STEMS:
            for ent in viewvalues(custom_entities):
                ent[UTTERANCES] = _stem_entity_utterances(
                    ent[UTTERANCES], language, resources)
        elif parser_usage is None:
            raise ValueError("A parser usage must be defined in order to fit "
                             "a CustomEntityParser")
        configuration = _create_custom_entity_parser_configuration(
            custom_entities)
        parser = GazetteerEntityParser.build(configuration)
        return cls(parser, language, parser_usage)


def _stem_entity_utterances(entity_utterances, language, resources):
    values = dict()
    # Sort by resolved value, so that values conflict in a deterministic way
    for raw_value, resolved_value in sorted(
            iteritems(entity_utterances), key=operator.itemgetter(1)):
        stemmed_value = stem(raw_value, language, resources)
        if stemmed_value not in values:
            values[stemmed_value] = resolved_value
    return values


def _merge_entity_utterances(raw_utterances, stemmed_utterances):
    # Sort by resolved value, so that values conflict in a deterministic way
    for raw_stemmed_value, resolved_value in sorted(
            iteritems(stemmed_utterances), key=operator.itemgetter(1)):
        if raw_stemmed_value not in raw_utterances:
            raw_utterances[raw_stemmed_value] = resolved_value
    return raw_utterances


def _create_custom_entity_parser_configuration(entities):
    return {
        "entity_parsers": [
            {
                "entity_identifier": entity_name,
                "entity_parser": {
                    "threshold": entity[MATCHING_STRICTNESS],
                    "gazetteer": [
                        {
                            "raw_value": k,
                            "resolved_value": v
                        } for k, v in sorted(iteritems(entity[UTTERANCES]))
                    ]
                }
            } for entity_name, entity in sorted(iteritems(entities))
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
