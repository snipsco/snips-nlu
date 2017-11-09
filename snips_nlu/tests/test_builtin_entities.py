from __future__ import unicode_literals

import traceback as tb
import unittest

import rustling
from builtin_entities_ontology import get_ontology
from mock import patch

from snips_nlu.builtin_entities import (
    get_builtin_entities, BuiltInEntity, scope_to_dim_kinds,
    RUSTLING_ENTITIES, RustlingParser)
from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY
from snips_nlu.languages import Language


class TestBuiltInEntities(unittest.TestCase):
    def test_rustling_parser_should_parse(self):
        # Given
        language = Language.EN
        text = "we'll be 2 at the meeting"
        parser = RustlingParser(language)

        # When / Then
        parse = parser.parse(text)

        expected_parse = [
            {
                "char_range": {
                    "start": 9, "end": 10
                }, "dim": "Number",
                "latent": False,
                "value": {
                    "type": "value",
                    "value": 2
                }
            }
        ]
        self.assertEqual(parse, expected_parse)

    @patch("snips_nlu.builtin_entities.RustlingParser.parse")
    def test_get_builtin_entities(self, mocked_parse):
        # Given
        language = "en"
        language = Language.from_iso_code(language)
        text = "nothing here but hey"
        mocked_parse.return_value = [
            {
                "char_range": {"start": 11, "end": 18},
                "value": {
                    'grain': 'hour',
                    'type': 'value',
                    'precision': 'exact',
                    'value': '2017-05-24 14:00:00 +02:00'
                },
                "dim": "Time"
            }
        ]

        # When
        expected_entities = [
            {
                MATCH_RANGE: (11, 18),
                VALUE: {
                    'grain': 'hour',
                    'type': 'value',
                    'precision': 'exact',
                    'value': '2017-05-24 14:00:00 +02:00'
                },
                ENTITY: BuiltInEntity.DATETIME
            }
        ]
        entities = get_builtin_entities(text, language)
        self.assertEqual(expected_entities, entities)

    def test_scope_to_dim_kinds(self):
        # Given
        scope = [entity for entity in BuiltInEntity]
        expected_dim_kinds = [
            "Time",
            "Number",
            "Ordinal",
            "AmountOfMoney",
            "Temperature",
            "Duration"
        ]

        # When
        dims = scope_to_dim_kinds(scope)

        # Then
        self.assertItemsEqual(expected_dim_kinds, dims)

    def test_built_in_label_uniqueness(self):
        # Given
        labels = [ent.value["label"] for ent in BuiltInEntity]

        # When
        unique_labels = set(labels)

        # Then
        self.assertEqual(len(unique_labels), len(labels))

    def test_built_in_label_rustling_dim_mapping(self):
        # Given
        rustling_names = [ent.value["rustling_dim_kind"]
                          for ent in BuiltInEntity]

        # When
        unique_rustling_names = set(rustling_names)

        # Then
        self.assertEqual(len(rustling_names), len(unique_rustling_names))

    def test_builtins_should_have_exactly_ontology_entities(self):
        # Given
        ontology = get_ontology()
        ontology_entities = [e["label"] for e in ontology["entities"]]

        # When
        entities = [e.label for e in BuiltInEntity]

        # Then
        self.assertItemsEqual(ontology_entities, entities)

    def test_entities_rustling_dim_kinds_should_exist(self):
        # Given
        supported_dim_kinds_by_language = rustling.all_configs()

        for ent in BuiltInEntity:
            # When / Then
            for language in ent.supported_languages:
                language_entities = supported_dim_kinds_by_language[
                    language.rustling_code]
                if ent.rustling_dim_kind not in language_entities:
                    self.fail(
                        "Unknown Rustling dimension '%s'" % ent.rustling_dim)

    def test_rustling_dim_kinds_should_exist(self):
        # Given
        supported_dim_kinds_by_language = rustling.all_configs()
        kinds = RUSTLING_ENTITIES

        for k in kinds:
            # When / Then
            if not any((k.rustling_dim_kind in language_dim_kinds)
                       for language_dim_kinds
                       in supported_dim_kinds_by_language.values()):
                self.fail("Unknown Rustling dimension kind '%s'" % k)

    def test_get_builtin_entities_should_support_all_languages(self):
        # Given
        text = ""

        for l in Language:
            # When / Then
            try:
                get_builtin_entities(text, l)
            except:  #pylint: disable=W0702
                self.fail("get_builtin_entities does not support %s."
                          "\n%s" % (l.iso_code, tb.format_exc()))

    def test_get_builtin_entities_should_respect_scope(self):
        # Given
        language = Language.EN
        text = "meet me at 10 p.m."

        # When
        scope = [BuiltInEntity.NUMBER]
        parse = get_builtin_entities(text, language, scope=scope)

        # Then
        self.assertEqual(len(parse), 1)
        self.assertEqual(parse[0][ENTITY], BuiltInEntity.NUMBER)
