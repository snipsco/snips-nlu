from __future__ import unicode_literals

import logging

from future.builtins import object, str
from future.utils import iteritems
from mock import MagicMock

from snips_nlu import __model_version__, __version__
from snips_nlu.common.dict_utils import LimitedSizeDict
from snips_nlu.common.log_utils import DifferedLoggingMessage
from snips_nlu.common.utils import (
    ranges_overlap, replace_entities_with_placeholders, parse_version)
from snips_nlu.constants import END, START
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.tests.utils import SnipsTest


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


class TestVersion(SnipsTest):
    def test_should_parse_valid_versions(self):
        # Given
        valid_versions = [
            "0.0.4.18",  # Version with subpatch number
            "1.0.4.18",  # Version with subpatch number
            "0.0.4",
            "0.0.4.0",
            "1.2.3",
            "10.20.30",
            "1.1.2-prerelease+meta",
            "1.1.2+meta",
            "1.1.2+meta-valid",
            "1.0.0-alpha",
            "1.0.0-beta",
            "1.0.0-alpha.beta",
            "1.0.0-alpha.beta.1",
            "1.0.0-alpha.1",
            "1.0.0-alpha0.valid",
            "1.0.0-alpha.0valid",
            "1.0.0-alpha-a.b-c-somethinglong+build.1-aef.1-its-okay",
            "1.0.0-rc.1+build.1",
            "2.0.0-rc.1+build.123",
            "1.2.3-beta",
            "10.2.3-DEV-SNAPSHOT",
            "1.2.3-SNAPSHOT-123",
            "1.0.0",
            "2.0.0",
            "1.1.7",
            "2.0.0+build.1848",
            "2.0.1-alpha.1227",
            "1.0.0-alpha+beta",
            "1.2.3----RC-SNAPSHOT.12.9.1--.12+788",
            "1.2.3----R-S.12.9.1--.12+meta",
            "1.2.3----RC-SNAPSHOT.12.9.1--.12",
            "1.0.0+0.build.1-rc.10000aaa-kk-0.1",
            "99999999999999999999999.999999999999999999.99999999999999999",
            "1.0.0-0A.is.legal",
        ]

        for v in valid_versions:
            msg = "Failed to parser valid version '%s'" % v
            with self.fail_if_exception(msg):
                # When / Then
                parse_version(v)

    def test_should_raise_on_invalid_versions(self):
        # Given
        invalid_versions = [
            "1",
            "1.2",
            "1.2.3-0123",
            "1.2.3-0123.0123",
            "1.1.2+.123",
            "+invalid",
            "-invalid",
            "-invalid+invalid",
            "-invalid.01",
            "alpha",
            "alpha.beta",
            "alpha.beta.1",
            "alpha.1",
            "alpha+beta",
            "alpha_beta",
            "alpha.",
            "alpha..",
            "beta",
            "1.0.0-alpha_beta",
            "-alpha.",
            "1.0.0-alpha..",
            "1.0.0-alpha..1",
            "1.0.0-alpha...1",
            "1.0.0-alpha....1",
            "1.0.0-alpha.....1",
            "1.0.0-alpha......1",
            "1.0.0-alpha.......1",
            "01.1.1",
            "1.01.1",
            "1.1.01",
            "1.2",
            "1.2.3.DEV",
            "1.2-SNAPSHOT",
            "1.2.31.2.3----RC-SNAPSHOT.12.09.1--..12+788",
            "1.2-RC-SNAPSHOT",
            "-1.0.3-gamma+b7718",
            "+justmeta",
            "9.8.7+meta+meta",
            "9.8.7-whatever+meta+meta",
            "99999999999999999999999.999999999999999999.99999999999999999"
            "----RC-SNAPSHOT.12.09.1--------------------------------..12",
        ]

        for v in invalid_versions:
            with self.assertRaises(ValueError) as exc:
                # When
                parse_version(v)
                # Then
                self.assertTrue(str(exc).startswith("Invalid version:"))

    def test_should_get_version_attributes(self):
        # Given
        v = "0.19.0.1-alpha1+meta"

        # When
        parsed_version = parse_version(v)

        # Then
        expected_parsed_version = {
            "major": "0",
            "minor": "19",
            "patch": "0",
            "subpatch": "1",
            "prerelease": "alpha1",
            "buildmetadata": "meta",
        }
        self.assertDictEqual(parsed_version, expected_parsed_version)

    def test_version_should_be_semantic(self):
        # Given
        v = __version__

        # When/Then
        msg = "Version number '%s' is not semantically valid" % v
        with self.fail_if_exception(msg):
            parse_version(v)

    def test_model_version_should_be_semantic(self):
        # Given
        model_version = __model_version__

        # When/Then
        msg = "Version number '%s' is not semantically valid" % model_version
        with self.fail_if_exception(msg):
            parse_version(model_version)
