import unittest

from mock import patch

from snips_nlu.built_in_entities import (get_built_in_entities, BuiltInEntity,
                                         scope_to_dims, clear_cache)
from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY
from snips_nlu.languages import Language


class TestBuiltInEntities(unittest.TestCase):
    @patch("duckling.core.parse")
    def test_get_built_in_entities(self, mocked_duckling_parse):
        # Given
        language = "en"
        language = Language.from_iso_code(language)
        text = "let's meet at 2p.m in the bronx"

        def mocked_parse(module, text):
            return [{
                'body': u'at 2p.m.',
                'dim': 'time',
                'end': 17,
                'value': {
                    'values': [{'grain': 'hour', 'type': 'value',
                                'value': '2017-03-29 14:00:00'},
                               {'grain': 'hour', 'type': 'value',
                                'value': '2017-03-30 14:00:00'},
                               {'grain': 'hour', 'type': 'value',
                                'value': '2017-03-31 14:00:00'}],
                    'type': 'value',
                    'grain': 'hour', 'value': '2017-03-27 14:00:00'},
                'start': 9}]

        mocked_duckling_parse.side_effect = mocked_parse
        expected_entities = [{MATCH_RANGE: (9, 17), VALUE: u'at 2p.m.',
                              ENTITY: BuiltInEntity.DATETIME}]

        # When
        entities = get_built_in_entities(text, language)
        self.assertEqual(expected_entities, entities)

    def test_scope_to_dims(self):
        # Given
        scope = [BuiltInEntity.DATETIME, BuiltInEntity.NUMBER]
        expected_dims = ["time", "number"]

        # When
        dims = scope_to_dims(scope)

        # Then
        self.assertEqual(expected_dims, dims)

    def test_built_in_label_uniqueness(self):
        # Given
        labels = [ent.value["label"] for ent in BuiltInEntity]

        # When
        unique_labels = set(labels)

        # Then
        self.assertEqual(len(unique_labels), len(labels))

    def test_built_in_label_duckling_dim_mapping(self):
        # Given
        duckling_names = [ent.value["duckling_dim"] for ent in BuiltInEntity]

        # When
        unique_duckling_name = set(duckling_names)

        # Then
        self.assertEqual(len(duckling_names), len(unique_duckling_name))

    @patch("duckling.core.parse")
    def test_duckling_cache(self, mocked_duckling_parse):
        # Given
        clear_cache()
        language = "en"
        language = Language.from_iso_code(language)
        text = "input text used twice"

        def mocked_parse(module, text):
            return []

        mocked_duckling_parse.side_effect = mocked_parse

        # When
        get_built_in_entities(text, language)
        get_built_in_entities(text, language)

        # Then
        mocked_duckling_parse.assert_called_once_with(language.duckling_code,
                                                      text)
