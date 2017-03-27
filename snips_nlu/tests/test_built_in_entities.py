import unittest

from duckling.core import default_context
from mock import patch

from snips_nlu.built_in_entities import get_built_in_entities, BuiltInEntity, \
    scope_to_dims


class TestBuiltInEntities(unittest.TestCase):
    @patch("duckling.core.parse")
    def test_get_built_in_entities(self, mocked_duckling_parse):
        # Given
        language = "en"
        text = "let's meet at 2p.m in the bronx"

        def mocked_parse(module, text, dims=[],
                         context=default_context("now")):
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
        expected_entities = [{"match_range": (9, 17), "value": u'at 2p.m.',
                              "entity": BuiltInEntity.DATETIME}]

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
