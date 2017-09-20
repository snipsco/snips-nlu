# coding=utf-8
from __future__ import unicode_literals

import unittest

from snips_nlu.languages import Language
from snips_nlu.tokenization import tokenize, Token, tokenize_light


class TestTokenization(unittest.TestCase):
    def test_should_tokenize_empty_string(self):
        # Given
        language = Language.EN
        text = ""

        # When
        tokens = tokenize(text, language)

        # Then
        self.assertListEqual(tokens, [])

    def test_should_tokenize_only_white_spaces(self):
        # Given
        text = "    "
        language = Language.EN

        # When
        tokens = tokenize(text, language)

        # Then
        self.assertListEqual(tokens, [])

    def test_should_tokenize_literals(self):
        # Given
        language = Language.EN
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
        language = Language.EN
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

    def test_should_tokenize_chinese(self):
        # Given
        language = Language.ZH
        s = "我要去乌克兰，我能知道那边的天气预报吗!"

        # When
        tokens = tokenize(s, language)

        # Then
        expected_tokens = [
            Token("我要", 0, 2),
            Token("去", 2, 3),
            Token("乌克兰", 3, 6),
            Token("我能", 7, 9),
            Token("知道", 9, 11),
            Token("那边", 11, 13),
            Token("的", 13, 14),
            Token("天气预报", 14, 18),
            Token("吗", 18, 19)
        ]
        self.assertSequenceEqual(tokens, expected_tokens)

    def test_should_tokenize_light_chinese(self):
        # Given
        language = Language.ZH
        s = "我要去乌克兰，我能知道那的天气预报吗!"

        # When
        tokenized_light = tokenize_light(s, language)

        # Then
        expected_s = ["我要", "去", "乌克兰", "我能", "知道", "那", "的", "天气预报",
                      "吗"]
        self.assertEqual(tokenized_light, expected_s)
