# coding=utf-8
from __future__ import unicode_literals

from snips_nlu_ontology import get_all_languages

from snips_nlu.constants import LANGUAGE_EN
from snips_nlu.preprocessing import Token, tokenize
from snips_nlu.tests.utils import SnipsTest


class TestPreprocessing(SnipsTest):
    def test_should_tokenize_empty_string(self):
        # Given
        language = LANGUAGE_EN
        text = ""

        # When
        tokens = tokenize(text, language)

        # Then
        self.assertTupleEqual(tokens, tuple())

    def test_should_tokenize_only_white_spaces(self):
        # Given
        text = "    "
        language = LANGUAGE_EN

        # When
        tokens = tokenize(text, language)

        # Then
        self.assertTupleEqual(tokens, tuple())

    def test_should_tokenize_literals(self):
        # Given
        language = LANGUAGE_EN
        text = "Hello Beautiful World"

        # When
        tokens = tokenize(text, language)

        # Then
        expected_tokens = (
            Token(value='Hello', start=0, end=5),
            Token(value='Beautiful', start=6, end=15),
            Token(value='World', start=16, end=21)
        )
        self.assertTupleEqual(tokens, expected_tokens)

    def test_should_tokenize_symbols(self):
        # Given
        language = LANGUAGE_EN
        text = "$$ % !!"

        # When
        tokens = tokenize(text, language)

        # Then
        expected_tokens = (
            Token(value='$$', start=0, end=2),
            Token(value='%', start=3, end=4),
            Token(value='!!', start=5, end=7)
        )
        self.assertTupleEqual(tokens, expected_tokens)

    def test_space_should_by_ignored(self):
        # Given
        text = " "
        for l in get_all_languages():
            # When
            tokens = tokenize(text, l)
            # Then
            self.assertEqual(len(tokens), 0)
