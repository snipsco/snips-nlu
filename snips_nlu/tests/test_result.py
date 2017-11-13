from __future__ import unicode_literals

import json
import unittest

from snips_nlu.constants import (PARSED_INTENT, PARSED_SLOTS, TEXT,
                                 INTENT_NAME, PROBABILITY, MATCH_RANGE,
                                 SLOT_NAME, VALUE)
from snips_nlu.result import Result, IntentClassificationResult, ParsedSlot


class TestResult(unittest.TestCase):
    def test_should_serialize_results(self):
        # Given
        result = Result(text="hello world",
                        parsed_intent=IntentClassificationResult("world", 0.5),
                        parsed_slots=[
                            ParsedSlot((3, 5), "slot_value", "slot_entity",
                                       "slot_name")])
        # When
        result_dict = result.as_dict()

        # Then
        try:
            json.dumps(result_dict)
        except:  # pylint: disable=W0702
            self.fail("Result dict should be json serializable")

        expected_dict = {
            PARSED_INTENT: {INTENT_NAME: 'world', PROBABILITY: 0.5},
            PARSED_SLOTS: [{MATCH_RANGE: [3, 5],
                            SLOT_NAME: 'slot_name',
                            VALUE: 'slot_value'}],
            TEXT: 'hello world'}
        self.assertDictEqual(result_dict, expected_dict)

    def test_should_serialize_results_when_none_values(self):
        # Given
        result = Result(text="hello world", parsed_intent=None,
                        parsed_slots=None)

        # When
        result_dict = result.as_dict()

        # Then
        expected_dict = {
            PARSED_INTENT: None,
            PARSED_SLOTS: None,
            TEXT: 'hello world'
        }
        self.assertDictEqual(result_dict, expected_dict)
