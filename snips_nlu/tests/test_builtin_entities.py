from __future__ import unicode_literals

import unittest

from builtin_entities_ontology import get_ontology
from mock import patch
from rustling import RustlingError
from rustling import RustlingParser as _RustlingParser

from snips_nlu.builtin_entities import (
    get_builtin_entities, BuiltInEntity, scope_to_dim_kinds,
    RUSTLING_DIM_KINDS, _RUSTLING_PARSERS, RustlingParser)
from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY
from snips_nlu.languages import Language


class TestBuiltInEntities(unittest.TestCase):
    def test_rustling_parser_should_parse(self):
        # Given
        language = Language.EN
        text = "let's meet at 2p.m in the bronx"
        parser = RustlingParser(language)

        expected_parse = [
            {
                "char_range": {"start": 11, "end": 18},
                "value": {
                    "grain": "hour",
                    "type": "value",
                    "precision": "exact",
                    "value": "2017-05-24 14:00:00 +02:00"
                },
                "latent": False,
                "dim": "time"
            }
        ]

        # When
        parse = parser.parse(text)
            

        # Then
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
                "dim": "time"
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
        scope = [BuiltInEntity.DATETIME, BuiltInEntity.NUMBER]
        expected_dim_kinds = ["time", "number"]

        # When
        dims = scope_to_dim_kinds(scope)

        # Then
        self.assertEqual(expected_dim_kinds, dims)

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
        text = "dummy text"
        language = "EN"
        parser = _RustlingParser(language)

        for ent in BuiltInEntity:
            # When / Then
            try:
                parser.parse_with_kind_order(
                    text, [ent.rustling_dim_kind.title()])
            except RustlingError():
                self.fail("Unknown Rustling dimension '%s'" % ent.rustling_dim)

    def test_rustling_dim_kinds_should_exist(self):
        # Given
        kinds = RUSTLING_DIM_KINDS
        text = "dummy text"
        language = "EN"
        parser = _RustlingParser(language)

        for k in kinds:
            # When / Then
            try:
                parser.parse_with_kind_order(text, [k.title()])
            except RustlingError:
                self.fail("Unknown Rustling dimension kind '%s'" % k)

    def test_get_builtin_entities_should_support_all_languages(self):
        # Given
        text = ""

        for l in Language:
            # When / Then
            try:
                get_builtin_entities(text, l)
            except:
                self.fail("get_builtin_entities does not support %s"
                          % l.iso_code)

    def test_builtin_entities_supported_languages_should_be_supported(self):
        # Given
        text = ""

        for entity in BuiltInEntity:
            for language in entity.supported_languages:
                if language in _RUSTLING_PARSERS:
                    # When / Then
                    try:
                        rust_parser = _RUSTLING_PARSERS[language].parser
                        rust_parser.parse_with_kind_order(
                            text, [entity.rustling_dim_kind.title()])
                    except RustlingError:
                        self.fail("Built in entity '%s' has '%s' as supported "
                                  "language, this dims kind is not supported "
                                  "in rustling")
