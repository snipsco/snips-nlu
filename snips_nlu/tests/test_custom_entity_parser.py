# coding=utf-8
from __future__ import unicode_literals

from pathlib import Path

from mock import patch

from snips_nlu.constants import STEMS
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.entity_parser import CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser import (
    CustomEntityParserUsage, _compute_char_shifts,
    _create_custom_entity_parser_configuration)
from snips_nlu.preprocessing import tokenize
from snips_nlu.tests.utils import FixtureTest

DATASET = validate_and_format_dataset({
    "intents": {

    },
    "entities": {
        "dummy_entity_1": {
            "data": [
                {
                    "value": "dummy_entity_1",
                    "synonyms": ["dummy_1"]
                }
            ],
            "use_synonyms": True,
            "automatically_extensible": True,
            "matching_strictness": 1.0,
            "license_info": {
                "filename": "LICENSE",
                "content": "some license content here"
            }
        },
        "dummy_entity_2": {
            "data": [
                {
                    "value": "dummy_entity_2",
                    "synonyms": ["dummy_2"]
                }
            ],
            "use_synonyms": True,
            "automatically_extensible": True,
            "matching_strictness": 1.0
        }
    },
    "language": "en"
})


class TestCustomEntityParser(FixtureTest):
    def test_should_parse_without_stems(self):
        # Given
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS, resources=dict())
        text = "dummy_entity_1 dummy_1 dummy_entity_2 dummy_2"

        # When
        result = parser.parse(text)
        result = sorted(result, key=lambda e: e["range"]["start"])

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 14
                },
                "entity_kind": "dummy_entity_1"
            },
            {
                "value": "dummy_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 15,
                    "end": 22
                },
                "entity_kind": "dummy_entity_1"
            },
            {
                "value": "dummy_entity_2",
                "resolved_value": "dummy_entity_2",
                "range": {
                    "start": 23,
                    "end": 37
                },
                "entity_kind": "dummy_entity_2"
            },
            {
                "value": "dummy_2",
                "resolved_value": "dummy_entity_2",
                "range": {
                    "start": 38,
                    "end": 45
                },
                "entity_kind": "dummy_entity_2"
            }
        ]
        self.assertListEqual(expected_entities, result)

    def test_should_parse_with_stems(self):
        # Given
        resources = {
            STEMS: {
                "dummy_entity_1": "dummy_entity_",
                "dummy_1": "dummy_"
            }
        }
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITH_STEMS, resources)
        text = "dummy_entity_ dummy_1"
        scope = ["dummy_entity_1"]

        # When
        result = parser.parse(text, scope=scope)

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 13
                },
                "entity_kind": "dummy_entity_1"
            }
        ]
        self.assertListEqual(expected_entities, result)

    def test_should_parse_with_and_without_stems(self):
        # Given
        resources = {STEMS: {"dummy_entity_1": "dummy_entity_"}}
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITH_AND_WITHOUT_STEMS, resources)
        scope = ["dummy_entity_1"]
        text = "dummy_entity_ dummy_1"

        # When
        result = parser.parse(text, scope=scope)

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 13
                },
                "entity_kind": "dummy_entity_1"
            },
            {
                "value": "dummy_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 14,
                    "end": 21
                },
                "entity_kind": "dummy_entity_1"
            }
        ]
        self.assertListEqual(expected_entities, result)

    def test_should_parse_with_proper_tokenization(self):
        # Given
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS, resources=dict())
        text = "  dummy_1?dummy_2"

        # When
        result = parser.parse(text)
        result = sorted(result, key=lambda e: e["range"]["start"])

        # Then
        expected_entities = [
            {
                "value": "dummy_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 2,
                    "end": 9
                },
                "entity_kind": "dummy_entity_1"
            },
            {
                "value": "dummy_2",
                "resolved_value": "dummy_entity_2",
                "range": {
                    "start": 10,
                    "end": 17
                },
                "entity_kind": "dummy_entity_2"
            }
        ]
        self.assertListEqual(expected_entities, result)

    def test_should_respect_scope(self):
        # Given
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS, resources=dict())
        scope = ["dummy_entity_1"]
        text = "dummy_entity_2"

        # When
        result = parser.parse(text, scope=scope)

        # Then
        self.assertListEqual([], result)

    @patch("snips_nlu_parsers.GazetteerEntityParser.parse")
    def test_should_use_cache(self, mocked_parse):
        # Given
        mocked_parse.return_value = []
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS, resources=dict())

        text = ""

        # When
        parser.parse(text)
        parser.parse(text)

        # Then
        self.assertEqual(1, mocked_parse.call_count)

    def test_should_be_serializable(self):
        # Given
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS, resources=dict())
        self.tmp_file_path.mkdir()
        parser_path = self.tmp_file_path / "custom_entity_parser"
        parser.persist(parser_path)
        loaded_parser = CustomEntityParser.from_path(parser_path)

        # When
        scope = ["dummy_entity_1"]
        text = "dummy_entity_1 dummy_1"
        result = loaded_parser.parse(text, scope=scope)

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 14
                },
                "entity_kind": "dummy_entity_1"
            },
            {
                "value": "dummy_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 15,
                    "end": 22
                },
                "entity_kind": "dummy_entity_1"
            }
        ]
        self.assertListEqual(expected_entities, result)
        license_path = parser_path / "parser" / "parser_1" / "LICENSE"
        self.assertTrue(license_path.exists())
        with license_path.open(encoding="utf8") as f:
            license_content = f.read()
        self.assertEqual("some license content here", license_content)

    def test_should_compute_tokenization_shift(self):
        # Given
        text = "  hello?   world"
        tokens = tokenize(text, "en")

        # When
        shifts = _compute_char_shifts(tokens)

        # Then
        expected_shifts = [-2, -2, -2, -2, -2, -1, -1, -3, -3, -3, -3, -3, -3]
        self.assertListEqual(expected_shifts, shifts)

    def test_create_custom_entity_parser_configuration(self):
        # Given
        entities = {
            "a": {
                "utterances":
                    {
                        "a a": "a",
                        "aa": "a",
                        "c": "c"
                    },
                "matching_strictness": 1.0
            },
            "b": {
                "utterances": {
                    "b": "b"
                },
                "matching_strictness": 1.0
            },
        }

        # When
        config = _create_custom_entity_parser_configuration(
            entities, stopwords_fraction=.5, language="en")

        # Then
        expected_dict = {
            "entity_parsers": [
                {
                    "entity_identifier": "a",
                    "entity_parser": {
                        "threshold": 1.0,
                        "n_gazetteer_stop_words": 1,
                        "gazetteer": [
                            {
                                "raw_value": "a a",
                                "resolved_value": "a",
                            },
                            {
                                "raw_value": "aa",
                                "resolved_value": "a",
                            },
                            {
                                "raw_value": "c",
                                "resolved_value": "c",
                            },
                        ]
                    }
                },
                {
                    "entity_identifier": "b",
                    "entity_parser": {
                        "threshold": 1.0,
                        "n_gazetteer_stop_words": 0,
                        "gazetteer": [
                            {
                                "raw_value": "b",
                                "resolved_value": "b",
                            },
                        ]
                    }
                }
            ]
        }
        self.assertDictEqual(expected_dict, config)


def _persist_parser(path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("nothing interesting here")


def _load_parser(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


# pylint: disable=unused-argument
def _stem(string, language):
    return string[:-1]

# pylint: enable=unused-argument
