from __future__ import unicode_literals

import logging

from future.builtins import object, str
from future.utils import iteritems
from mock import MagicMock

from snips_nlu.constants import START, END
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.tests.utils import SnipsTest
from snips_nlu.common.utils import (
    ranges_overlap, replace_entities_with_placeholders)
from snips_nlu.common.log_utils import DifferedLoggingMessage
from snips_nlu.common.dict_utils import LimitedSizeDict


class TestLimitedSizeDict(SnipsTest):
    def test_should_raise_when_no_size_limit(self):
        # Given/When/Then
        with self.assertRaises(ValueError) as ctx:
            LimitedSizeDict()
        self.assertEqual(str(ctx.exception.args[0]),
                         "'size_limit' must be passed as a keyword argument")

    def test_should_initialize_with_argument(self):
        # Given
        sequence = [("a", 1), ("b", 2)]
        size_limit = 3
        # When
        d = LimitedSizeDict(sequence, size_limit=size_limit)
        # Then
        items = sorted(iteritems(d), key=lambda i: i[0])
        self.assertListEqual(items, sequence)

    def test_should_initialize_without_argument(self):
        # Given
        size_limit = 10
        # When
        d = LimitedSizeDict(size_limit=size_limit)
        # Then
        self.assertListEqual(list(d), [])

    def test_should_wrong_when_initialization_should_raise_error(self):
        # Given
        sequence = [("a", 1), ("b", 2), ("c", 3)]
        size_limit = len(sequence) - 1
        # When/Then
        with self.assertRaises(ValueError) as ctx:
            LimitedSizeDict(sequence, size_limit=size_limit)
        self.assertEqual(str(ctx.exception.args[0]),
                         "Tried to initialize LimitedSizedDict with more "
                         "value than permitted with 'limit_size'")

    def test_should_erase_items_when_updating(self):
        # Given
        sequence = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        size_limit = len(sequence) - 2
        # When
        my_dict = LimitedSizeDict(sequence[:size_limit], size_limit=size_limit)
        for k, v in sequence[size_limit:]:
            my_dict[k] = v
        # Then
        items = sorted(list(iteritems(my_dict)), key=lambda i: i[0])
        self.assertListEqual(items, sequence[size_limit:])


class TestUtils(SnipsTest):
    def test_ranges_overlap(self):
        # Given
        range1 = [4, 8]
        range2 = [5, 7]
        range3 = [3, 9]
        range4 = [3, 4]
        range5 = [8, 9]
        range6 = [3, 6]
        range7 = [4, 10]

        # When / Then
        self.assertTrue(ranges_overlap(range1, range2))
        self.assertTrue(ranges_overlap(range1, range3))
        self.assertFalse(ranges_overlap(range1, range4))
        self.assertFalse(ranges_overlap(range1, range5))
        self.assertTrue(ranges_overlap(range1, range6))
        self.assertTrue(ranges_overlap(range1, range7))

    def test_differed_logging_message(self):
        # Given
        def fn(a, b, c):
            return a + b + c

        mocked_fn = MagicMock()
        mocked_fn.side_effect = fn

        class Greeter(object):
            def greet(self):
                return "Yo!"

        levels = [logging.DEBUG, logging.INFO, logging.WARNING]
        logger = logging.Logger("my_dummy_logger", logging.INFO)
        logger.addHandler(logging.StreamHandler())
        a_, b_, c_ = 1, 2, 3

        with self.fail_if_exception("Failed to log"):
            # When/Then
            my_obj = Greeter()
            logger.log(logging.INFO,
                       "Greeting: %s", DifferedLoggingMessage(my_obj.greet))
            for l in levels:
                logger.log(l, "Level: %s -> %s", str(l),
                           DifferedLoggingMessage(mocked_fn, a_, b_, c=c_))
        self.assertEqual(2, mocked_fn.call_count)

    def test_should_replace_entities(self):
        # Given
        text = "Be the first to be there at 9pm"

        # When
        entities = [
            {
                "entity_kind": "snips/ordinal",
                "value": "the first",
                "range": {
                    "start": 3,
                    "end": 12
                }
            },
            {
                "entity_kind": "my_custom_entity",
                "value": "first",
                "range": {
                    "start": 7,
                    "end": 12
                }
            },
            {
                "entity_kind": "snips/datetime",
                "value": "at 9pm",
                "range": {
                    "start": 25,
                    "end": 31
                }
            }
        ]

        def placeholder_fn(x):
            return "%%%s%%" % "".join(tokenize_light(x, "en")).upper()

        range_mapping, processed_text = replace_entities_with_placeholders(
            text=text, entities=entities, placeholder_fn=placeholder_fn)

        # Then
        expected_mapping = {
            (3, 17): {START: 3, END: 12},
            (30, 45): {START: 25, END: 31}
        }
        expected_processed_text = \
            "Be %SNIPSORDINAL% to be there %SNIPSDATETIME%"

        self.assertDictEqual(expected_mapping, range_mapping)
        self.assertEqual(expected_processed_text, processed_text)
