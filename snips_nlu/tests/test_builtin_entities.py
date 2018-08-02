from __future__ import unicode_literals

from mock import patch
from snips_nlu_ontology import get_all_languages

from snips_nlu.builtin_entities import (
    BuiltinEntityParser, _BUILTIN_ENTITY_PARSERS, get_builtin_entity_parser,
    get_builtin_entity_parser_from_scope)
from snips_nlu.constants import ENTITIES, ENTITY_KIND, LANGUAGE
from snips_nlu.tests.utils import SnipsTest


class TestBuiltInEntities(SnipsTest):
    def setUp(self):
        _BUILTIN_ENTITY_PARSERS.clear()

    def test_should_parse_grammar_entities(self):
        # Given
        text = "we'll be 2 at the meeting"
        parser = BuiltinEntityParser("en", None)

        # When / Then
        parse = parser.parse(text)

        expected_parse = [
            {
                "entity": {
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
        parser = get_builtin_entity_parser_from_scope("fr",
                                                      ["snips/musicArtist"])

        # When / Then
        parse = parser.parse(text)

        expected_parse = [
            {
                "entity": {
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
            parser = BuiltinEntityParser(language, None)
            msg = "get_builtin_entities does not support %s." % language
            with self.fail_if_exception(msg):
                # When / Then
                parser.parse(text)

    def test_should_respect_scope(self):
        # Given
        text = "meet me at 10 p.m."

        # When
        scope = ["snips/number"]
        parser = BuiltinEntityParser("en", None)
        parse = parser.parse(text, scope=scope)

        # Then
        self.assertEqual(len(parse), 1)
        self.assertEqual(parse[0][ENTITY_KIND], "snips/number")

    def test_should_respect_scope_with_gazetteer_entity(self):
        # Given
        text = "je veux écouter metallica"

        # When
        gazetteer_entities = ["snips/musicArtist", "snips/musicAlbum"]
        parser = get_builtin_entity_parser_from_scope("fr", gazetteer_entities)
        scope1 = ["snips/musicArtist"]
        parse1 = parser.parse(text, scope=scope1)
        scope2 = ["snips/musicAlbum"]
        parse2 = parser.parse(text, scope=scope2)

        # Then
        expected_parse1 = [
            {
                "entity": {
                    "kind": "MusicArtist",
                    "value": "Metallica"
                },
                "entity_kind": "snips/musicArtist",
                "range": {
                    "end": 25,
                    "start": 16
                },
                "value": "metallica"
            }
        ]
        expected_parse2 = [
            {
                "entity": {
                    "kind": "MusicAlbum",
                    "value": "Metallica"
                },
                "entity_kind": "snips/musicAlbum",
                "range": {
                    "end": 25,
                    "start": 16
                },
                "value": "metallica"
            }
        ]
        self.assertEqual(expected_parse1, parse1)
        self.assertEqual(expected_parse2, parse2)

    def test_should_not_disambiguate_grammar_and_gazetteer_entities(self):
        # Given
        text = "trois nuits par semaine"
        gazetteer_entities = ["snips/musicTrack"]
        parser = get_builtin_entity_parser_from_scope("fr", gazetteer_entities)

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
                "entity": {
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
                "entity": {
                    "kind": "MusicTrack",
                    "value": "3 nuits par semaine"
                },
                "entity_kind": "snips/musicTrack"
            }
        ]
        self.assertListEqual(expected_result, result)

    @patch("snips_nlu.builtin_entities.BuiltinEntityParser")
    @patch("snips_nlu.builtin_entities.find_gazetteer_entity_data_path")
    def test_should_share_parser(
            self, mocked_find_gazetteer_entity_data_path, mocked_parser):
        # Given
        dataset1 = {
            LANGUAGE: "en",
            ENTITIES: {
                "snips/musicArtist": {},
                "snips/musicTrack": {},
                "snips/number": {}
            }
        }

        dataset2 = {
            LANGUAGE: "en",
            ENTITIES: {
                "snips/musicTrack": {},
                "snips/musicAlbum": {},
                "snips/amountOfMoney": {}
            }
        }

        dataset3 = {
            LANGUAGE: "en",
            ENTITIES: {
                "snips/musicTrack": {},
                "snips/musicArtist": {},
            }
        }

        def mock_find_gazetteer_entity_data_path(language, entity_name):
            return "mocked_path"

        mocked_find_gazetteer_entity_data_path.side_effect = \
            mock_find_gazetteer_entity_data_path

        # When
        get_builtin_entity_parser(dataset1)
        get_builtin_entity_parser(dataset2)
        get_builtin_entity_parser(dataset3)

        # Then
        self.assertEqual(2, mocked_parser.call_count)
