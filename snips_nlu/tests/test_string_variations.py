# coding=utf-8
from __future__ import unicode_literals

from snips_nlu.builtin_entities import BuiltinEntityParser
from snips_nlu.constants import (LANGUAGE_EN, LANGUAGE_FR, RES_MATCH_RANGE,
                                 SNIPS_NUMBER, START)
from snips_nlu.string_variations import (
    alphabetic_value, and_variations, get_string_variations,
    numbers_variations, punctuation_variations)
from snips_nlu.tests.utils import SnipsTest


class TestStringVariations(SnipsTest):
    def test_and_variations(self):
        # Given
        language = LANGUAGE_EN
        data = [
            ("a and b and c",
             {
                 "a and b & c",
                 "a & b and c",
                 "a & b & c",
                 "a and b and c"
             }),
            ("random", set()),
            ("be&you", set())
        ]

        for string, expected_variations in data:
            # When
            variations = and_variations(string, language)
            # Then
            self.assertSetEqual(variations, expected_variations)

    def test_punctuation_variations(self):
        # Given
        language = LANGUAGE_EN
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
            ("random", set()),
        ]

        for string, expected_variations in data:
            # When
            variations = punctuation_variations(string, language)

            # Then
            self.assertSetEqual(variations, expected_variations)

    def test_alphabetic_value(self):
        # Given
        language = LANGUAGE_EN
        string = "1 time and 23 times and one thousand and sixty and 1.2"
        parser = BuiltinEntityParser(language, None)
        entities = parser.parse(string, scope=[SNIPS_NUMBER])
        entities = sorted(entities, key=lambda x: x[RES_MATCH_RANGE][START])

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
        language = LANGUAGE_EN
        string = "a and b 2"

        # When
        variations = get_string_variations(string, language,
                                           BuiltinEntityParser("en", None))

        # Then
        expected_variations = {
            "a and b 2",
            "a & b 2",
            "a b 2",
            "a and b two",
            "a & b two",
            "a b two",
            "A B two",
            "A And B two",
            "A  B 2",
            "A and B two",
            "A & B two",
            "A & B 2",
            "A  B two",
            "A and B 2",
            "a  b 2",
            "a  b two",
            "A B 2",
            "A And B 2",
        }
        self.assertSetEqual(variations, expected_variations)

    def test_should_variate_case_and_normalization(self):
        # Given
        language = LANGUAGE_EN
        string = "Küche"

        # When
        variations = get_string_variations(string, language,
                                           BuiltinEntityParser("en", None))

        # Then
        expected_variations = {
            "kuche",
            "küche",
            "Kuche",
            "Küche"
        }
        self.assertSetEqual(variations, expected_variations)

    def test_get_france_24(self):
        # Given
        language = LANGUAGE_FR
        string = "france 24"

        # When
        variations = get_string_variations(string, language,
                                           BuiltinEntityParser("en", None))

        # Then
        expected_variations = {
            "france vingt-quatre",
            "France vingt-quatre",
            "france vingt quatre",
            "France vingt quatre",
            "france 24",
            "France 24",
        }
        self.assertSetEqual(variations, expected_variations)

    def test_numbers_variations_should_handle_floats(self):
        # Given
        language = LANGUAGE_EN
        string = "7.62 mm caliber 2 and six"

        # When
        variations = numbers_variations(string, language,
                                        BuiltinEntityParser("en", None))

        # Then
        expected_variations = {
            "7.62 mm caliber 2 and six",
            "7.62 mm caliber two and six",
            "7.62 mm caliber 2 and 6",
            "7.62 mm caliber two and 6",
        }
        self.assertSetEqual(variations, expected_variations)
