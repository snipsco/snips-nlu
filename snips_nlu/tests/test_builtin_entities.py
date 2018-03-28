# coding=utf-8
from __future__ import unicode_literals

from mock import patch
from snips_nlu_ontology import get_all_languages

from snips_nlu.builtin_entities import (
    get_builtin_entities, BuiltinEntityParser)
from snips_nlu.constants import ENTITY_KIND
from snips_nlu.tests.utils import SnipsTest


class TestBuiltInEntities(SnipsTest):
    def test_builtin_entity_parser_should_parse(self):
        # Given
        text = "we'll be 2 at the meeting"
        parser = BuiltinEntityParser("en")

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

    def test_get_builtin_entities_should_support_all_languages(self):
        # Given
        text = ""

        for language in get_all_languages():
            msg = "get_builtin_entities does not support %s." % language
            with self.fail_if_exception(msg):
                # When / Then
                get_builtin_entities(text, language)

    def test_get_builtin_entities_should_respect_scope(self):
        # Given
        text = "meet me at 10 p.m."

        # When
        scope = ("snips/number",)
        parse = get_builtin_entities(text, "en", scope=scope)

        # Then
        self.assertEqual(len(parse), 1)
        self.assertEqual(parse[0][ENTITY_KIND], "snips/number")

    def test_should_get_entities_for_non_space_separated_languages(self):
        # Given
        language = "ja"
        text = " aaa  bbb  ccc  ddd  eee "
        with patch("snips_nlu.builtin_entities._BuiltinEntityParser.parse") \
                as mocked_parse:
            # joined text "aaabbbcccdddeee"
            mocked_parse.return_value = [
                {
                    "entity_kind": "dummy_entity",
                    "range": {
                        "start": 0,
                        "end": 6
                    },
                    "value": "aaabbb"
                },
                {
                    "entity_kind": "dummy_entity",
                    "range": {
                        "start": 8,
                        "end": 10
                    },
                    "value": "cd"
                },
                {
                    "entity_kind": "dummy_entity",
                    "range": {
                        "start": 12,
                        "end": 15
                    },
                    "value": "eee"
                }
            ]
            # When
            entities = get_builtin_entities(text, language)

            # Then
            expected_entities = [
                {
                    "entity_kind": "dummy_entity",
                    "range": {
                        "start": 1,
                        "end": 9
                    },
                    "value": "aaa  bbb"
                },
                {
                    "entity_kind": "dummy_entity",
                    "range": {
                        "start": 21,
                        "end": 24
                    },
                    "value": "eee"
                }
            ]
        self.assertSequenceEqual(expected_entities, entities)

    def test_builtin_entity_cache(self):
        # Given
        language = "en"
        cache_size = 48

        # When
        parser = BuiltinEntityParser(language, cache_size=cache_size)

        # Then
        self.assertTrue(hasattr(parser.parse, "cache_info"))
        self.assertEqual(cache_size, parser.parse.cache_info().maxsize)
