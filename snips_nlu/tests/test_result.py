from __future__ import unicode_literals

import json
import unittest

from snips_nlu.constants import (RES_INTENT, RES_SLOTS, RES_INTENT_NAME,
                                 RES_PROBABILITY, RES_MATCH_RANGE, RES_INPUT,
                                 RES_ENTITY, RES_SLOT_NAME, RES_VALUE)
from snips_nlu.result import (parsing_result, intent_classification_result,
                              unresolved_slot)


class TestResult(unittest.TestCase):
    def test_should_serialize_results(self):
        # Given
        input_ = "hello world"
        intent = intent_classification_result("world", 0.5)
        slots = [unresolved_slot([3, 5],
                                 "slot_value",
                                 "slot_entity",
                                 "slot_name")]

        # When
        result = parsing_result(input=input_, intent=intent, slots=slots)

        # Then
        msg = "Result dict should be json serializable"
        with self.fail_if_exception(msg):
            json.dumps(result)

        expected_result = {
            RES_INTENT: {RES_INTENT_NAME: 'world', RES_PROBABILITY: 0.5},
            RES_SLOTS: [{RES_MATCH_RANGE: {"start": 3, "end": 5},
                         RES_ENTITY: 'slot_entity',
                         RES_SLOT_NAME: 'slot_name',
                         RES_VALUE: 'slot_value'}],
            RES_INPUT: input_}
        self.assertDictEqual(expected_result, result)

    def test_should_serialize_results_when_none_values(self):
        # Given
        input_ = "hello world"

        # When
        result = parsing_result(input=input_, intent=None, slots=None)

        # Then
        expected_result = {
            RES_INTENT: None,
            RES_SLOTS: None,
            RES_INPUT: input_
        }
        self.assertDictEqual(expected_result, result)
