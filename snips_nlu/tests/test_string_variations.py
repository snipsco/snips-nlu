# coding=utf-8
from __future__ import unicode_literals

import unittest

from snips_nlu.builtin_entities import get_builtin_entities
from snips_nlu.constants import RES_MATCH_RANGE
from snips_nlu.languages import Language
from snips_nlu.string_variations import (
    and_variations, alphabetic_value, punctuation_variations,
    get_string_variations, numbers_variations)


class TestStringVariations(unittest.TestCase):
    def test_and_variations(self):
        # Given
        language = Language.EN
        data = [
            ("a and b and c",
             {
                 "a and b & c",
                 "a & b and c",
                 "a & b & c",
                 "a and b and c"
             }),
            ("random", {}),
            ("be&you", {})
        ]

        for string, expected_variations in data:
            # When
            variations = and_variations(string, language)
            # Then
            self.assertItemsEqual(variations, expected_variations)

    def test_punctuation_variations(self):
        # Given
        language = Language.EN
        data = [
            ("a! ?b .c",
             {
                 "a! ?b .c",
                 "a! b .c",
                 "a! ?b c",
                 "a! b c",
                 "a ?b .c",
                 "a b .c",
                 "a ?b c",
                 "a b c",
             }),
            ("random", {}),
        ]

        for string, expected_variations in data:
            # When
            variations = punctuation_variations(string, language)

            # Then
            self.assertItemsEqual(variations, expected_variations)

    def test_alphabetic_value(self):
        # Given
        language = Language.EN
        string = "1 time and 23 times and one thousand and sixty and 1.2"
        entities = get_builtin_entities(string, language)
        entities = sorted(entities, key=lambda x: x[RES_MATCH_RANGE])

        expected_values = ["one", "twenty-three", "one thousand and sixty",
                           None]

        self.assertEqual(len(entities), len(expected_values))

        for i, ent in enumerate(entities):
            # When
            value = alphabetic_value(ent, language)

            # Then
            self.assertEqual(value, expected_values[i])

    def test_get_string_variations(self):
        # Given
        language = Language.EN
        string = "a and b 2"

        # When
        variations = get_string_variations(string, language)

        # Then
        expected_variations = {
            "a and b 2",
            "a & b 2",
            "a b 2",
            "a and b two",
            "a & b two",
            "a b two"
        }
        self.assertItemsEqual(variations, expected_variations)

    def test_numbers_variations_should_handle_floats(self):
        # Given
        language = Language.EN
        string = "7.62 mm caliber 2 and six"

        # When
        variations = numbers_variations(string, language)

        # Then
        expected_variations = {
            "7.62 mm caliber 2 and six",
            "7.62 mm caliber two and six",
            "7.62 mm caliber 2 and 6",
            "7.62 mm caliber two and 6",
        }
        self.assertItemsEqual(variations, expected_variations)
