# coding=utf-8
from __future__ import unicode_literals

import unittest

from snips_nlu_ontology_rust import get_all_languages

from snips_nlu.constants import LANGUAGE_EN
from snips_nlu.tokenization import tokenize, Token


class TestTokenization(unittest.TestCase):
    def test_should_tokenize_empty_string(self):
        # Given
        language = LANGUAGE_EN
        text = ""

        # When
        tokens = tokenize(text, language)

        # Then
        self.assertListEqual(tokens, [])

    def test_should_tokenize_only_white_spaces(self):
        # Given
        text = "    "
        language = LANGUAGE_EN

        # When
        tokens = tokenize(text, language)

        # Then
        self.assertListEqual(tokens, [])

    def test_should_tokenize_literals(self):
        # Given
        language = LANGUAGE_EN
        text = "Hello Beautiful World"

        # When
        tokens = tokenize(text, language)

        # Then
        expected_tokens = [
            Token(value='Hello', start=0, end=5, stem=None),
            Token(value='Beautiful', start=6, end=15, stem=None),
            Token(value='World', start=16, end=21, stem=None)
        ]
        self.assertListEqual(tokens, expected_tokens)

    def test_should_tokenize_symbols(self):
        # Given
        language = LANGUAGE_EN
        text = "$$ % !!"

        # When
        tokens = tokenize(text, language)

        # Then
        expected_tokens = [
            Token(value='$$', start=0, end=2, stem=None),
            Token(value='%', start=3, end=4, stem=None),
            Token(value='!!', start=5, end=7, stem=None)
        ]
        self.assertListEqual(tokens, expected_tokens)

    def test_space_should_by_ignored(self):
        # Given
        text = " "
        for l in get_all_languages():
            # When
            tokens = tokenize(text, l)
            # Then
            self.assertEqual(len(tokens), 0)
