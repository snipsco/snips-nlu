# coding=utf-8
from __future__ import unicode_literals

from mock import patch
from snips_nlu_parsers import get_all_languages

from snips_nlu.constants import ENTITIES, LANGUAGE
from snips_nlu.entity_parser.builtin_entity_parser import (
    BuiltinEntityParser, _BUILTIN_ENTITY_PARSERS)
from snips_nlu.tests.utils import SnipsTest


class TestBuiltinEntityParser(SnipsTest):
    def setUp(self):
        _BUILTIN_ENTITY_PARSERS.clear()

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

    @patch("snips_nlu.entity_parser.builtin_entity_parser"
           "._build_builtin_parser")
    def test_should_share_parser(self, mocked_build_builtin_parser):
        # Given

        # pylint:disable=unused-argument
        def mock_build_builtin_parser(language, gazetteer_entity_scope):
            return None

        # pylint:enable=unused-argument

        mocked_build_builtin_parser.side_effect = mock_build_builtin_parser

        dataset1 = {
            LANGUAGE: "fr",
            ENTITIES: {
                "snips/musicArtist": {},
                "snips/musicTrack": {},
                "snips/number": {}
            }
        }

        dataset2 = {
            LANGUAGE: "fr",
            ENTITIES: {
                "snips/musicTrack": {},
                "snips/musicAlbum": {},
                "snips/amountOfMoney": {}
            }
        }

        dataset3 = {
            LANGUAGE: "fr",
            ENTITIES: {
                "snips/musicTrack": {},
                "snips/musicArtist": {},
            }
        }

        # When
        BuiltinEntityParser.build(dataset=dataset1)
        BuiltinEntityParser.build(dataset=dataset2)
        BuiltinEntityParser.build(dataset=dataset3)

        # Then
        self.assertEqual(2, mocked_build_builtin_parser.call_count)
