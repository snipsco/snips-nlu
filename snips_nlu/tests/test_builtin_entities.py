from __future__ import unicode_literals

from snips_nlu_ontology import get_all_languages

from snips_nlu.builtin_entities import BuiltinEntityParser
from snips_nlu.constants import ENTITY_KIND
from snips_nlu.tests.utils import SnipsTest


class TestBuiltInEntities(SnipsTest):
    def test_builtin_entity_parser_should_parse(self):
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

    def test_get_builtin_entities_should_support_all_languages(self):
        # Given
        text = ""

        for language in get_all_languages():
            parser = BuiltinEntityParser(language, None)
            msg = "get_builtin_entities does not support %s." % language
            with self.fail_if_exception(msg):
                # When / Then
                parser.parse(text)

    def test_get_builtin_entities_should_respect_scope(self):
        # Given
        text = "meet me at 10 p.m."

        # When
        scope = ["snips/number"]
        parser = BuiltinEntityParser("en", None)
        parse = parser.parse(text, scope=scope)

        # Then
        self.assertEqual(len(parse), 1)
        self.assertEqual(parse[0][ENTITY_KIND], "snips/number")
