# coding=utf-8
from __future__ import unicode_literals

import io

from snips_nlu_parsers import get_all_languages

from snips_nlu.dataset import Dataset, validate_and_format_dataset
from snips_nlu.entity_parser.builtin_entity_parser import BuiltinEntityParser
from snips_nlu.tests.utils import SnipsTest


class TestBuiltinEntityParser(SnipsTest):
    def test_should_parse_grammar_entities(self):
        # Given
        text = "we'll be 2 at the meeting"
        language = "en"
        parser = BuiltinEntityParser.build(language=language)

        # When / Then
        parse = parser.parse(text)

        expected_parse = [
            {
                "resolved_value": {
                    "kind": "Number",
                    "value": 2.0
                },
                "entity_kind": "snips/number",
                "range": {
                    "end": 10,
                    "start": 9
                },
                "value": "2"
            }
        ]
        self.assertEqual(parse, expected_parse)

    def test_should_parse_gazetteer_entities(self):
        # Given
        text = "je veux ecouter les daft punk s'il vous plait"
        parser = BuiltinEntityParser.build(
            language="fr", gazetteer_entity_scope=["snips/musicArtist"])

        # When / Then
        parse = parser.parse(text)

        expected_parse = [
            {
                "resolved_value": {
                    "kind": "MusicArtist",
                    "value": "Daft Punk"
                },
                "entity_kind": "snips/musicArtist",
                "range": {
                    "end": 29,
                    "start": 20
                },
                "value": "daft punk"
            }
        ]
        self.assertEqual(parse, expected_parse)

    def test_should_parse_extended_gazetteer_entities(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: play_artist
utterances:
  - I want to listen to [artist:snips/musicArtist](daft punk)
  - Play me a song by [artist](the stones)
  - Play [artist]
  - Could you play some [artist] please ?

---
type: entity
name: snips/musicArtist
extended_values:
- [my first custom artist, my first custom artist alias]
- my second custom artist""")

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        dataset = validate_and_format_dataset(dataset)
        parser = BuiltinEntityParser.build(dataset=dataset)

        # When / Then
        parse = parser.parse(
            "I want to listen to my first custom artist alias and my second "
            "custom artist as well")

        first_result = {
            "resolved_value": {
                "kind": "MusicArtist",
                "value": "my first custom artist"
            },
            "entity_kind": "snips/musicArtist",
            "range": {
                "end": 48,
                "start": 20
            },
            "value": "my first custom artist alias"
        }

        second_result = {
            "resolved_value": {
                "kind": "MusicArtist",
                "value": "my second custom artist"
            },
            "entity_kind": "snips/musicArtist",
            "range": {
                "end": 76,
                "start": 53
            },
            "value": "my second custom artist"
        }

        self.assertTrue(first_result in parse)
        self.assertTrue(second_result in parse)

    def test_should_support_all_languages(self):
        # Given
        text = ""

        for language in get_all_languages():
            parser = BuiltinEntityParser.build(language=language)
            msg = "get_builtin_entities does not support %s." % language
            with self.fail_if_exception(msg):
                # When / Then
                parser.parse(text)

    def test_should_not_disambiguate_grammar_and_gazetteer_entities(self):
        # Given
        text = "trois nuits par semaine"
        gazetteer_entities = ["snips/musicTrack"]
        parser = BuiltinEntityParser.build(
            language="fr", gazetteer_entity_scope=gazetteer_entities)

        # When
        result = parser.parse(text)

        # Then
        expected_result = [
            {
                "value": "trois",
                "range": {
                    "start": 0,
                    "end": 5
                },
                "resolved_value": {
                    "kind": "Number",
                    "value": 3.0
                },
                "entity_kind": "snips/number"
            },
            {
                "value": "trois nuits par semaine",
                "range": {
                    "start": 0,
                    "end": 23
                },
                "resolved_value": {
                    "kind": "MusicTrack",
                    "value": "3 nuits par semaine"
                },
                "entity_kind": "snips/musicTrack"
            }
        ]
        self.assertListEqual(expected_result, result)
