from __future__ import unicode_literals

from builtins import next, range

import numpy as np
from mock import patch

from snips_nlu.constants import LANGUAGE_EN
from snips_nlu.data_augmentation import (
    capitalize, capitalize_utterances, generate_utterance,
    get_contexts_iterator, get_entities_iterators)
from snips_nlu.tests.utils import SnipsTest


def np_random_permutation(x):
    return x


class TestDataAugmentation(SnipsTest):
    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    def test_context_iterator(self, _):
        # Given
        dataset = {
            "intents": {
                "dummy": {"utterances": list(range(3))}
            }
        }
        random_state = np.random.RandomState(1)

        # When
        it = get_contexts_iterator(dataset, "dummy", random_state)
        context = [next(it) for _ in range(5)]

        # Then
        self.assertEqual(context, [0, 2, 1, 0, 2])

    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    def test_entities_iterators(self, _):
        # Given
        language = "en"
        intent_entities = {
            "entity1": {
                "utterances": {
                    "entity 1": "entity 1",
                    "entity 11": "entity 11",
                    "entity 111": "entity 111",
                }
            },
            "entity2": {
                "utterances": {
                    "entity 2": "entity 2",
                    "entity 22": "entity 22",
                    "entity 222": "entity 222",
                }
            },
            "snips/number": {
                "utterances": {"two", "21"}
            }
        }
        random_state = np.random.RandomState(1)

        # Then
        it_dict = get_entities_iterators(intent_entities, language,
                                         False, random_state)

        # When
        self.assertIn("entity1", it_dict)
        expected_seq = ["entity 1", "entity 11", "entity 111"]
        seq = [next(it_dict["entity1"]) for _ in range(len(expected_seq))]
        self.assertListEqual(expected_seq, sorted(seq))

        self.assertIn("entity2", it_dict)
        expected_seq = ["entity 2", "entity 22", "entity 222"]
        seq = [next(it_dict["entity2"]) for _ in range(len(expected_seq))]
        self.assertListEqual(expected_seq, sorted(seq))

        self.assertIn("snips/number", it_dict)
        expected_seq = ["21", "two"]
        seq = [next(it_dict["snips/number"]) for _ in range(len(expected_seq))]
        self.assertListEqual(expected_seq, sorted(seq))

    @patch("snips_nlu.data_augmentation.get_builtin_entity_examples")
    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    def test_entities_iterators_with_builtin_examples(
            self, _, mocked_builtin_entity_examples):
        # Given
        language = "en"

        def mock_builtin_entity_examples(builtin_entity_kind, _):
            if builtin_entity_kind == "snips/number":
                return ["2007", "two hundreds and six"]
            return []

        mocked_builtin_entity_examples.side_effect = \
            mock_builtin_entity_examples

        intent_entities = {
            "entity1": {
                "utterances": {
                    "entity 1": "entity 1",
                    "entity 11": "entity 11",
                    "entity 111": "entity 111",
                }
            },
            "snips/number": {
                "utterances": {"9", "seventy"}
            }
        }
        random_state = np.random.RandomState(1)

        # Then
        add_builtin_entities_examples = True
        it_dict = get_entities_iterators(intent_entities, language,
                                         add_builtin_entities_examples,
                                         random_state)

        # When
        self.assertIn("entity1", it_dict)
        expected_seq = ["entity 1", "entity 11", "entity 111"]
        seq = [next(it_dict["entity1"]) for _ in range(len(expected_seq))]
        self.assertListEqual(expected_seq, sorted(seq))

        self.assertIn("snips/number", it_dict)
        # Check that entity examples are at the beginning of the iterator
        expected_seq_start = ["2007", "two hundreds and six"]
        seq_start = [next(it_dict["snips/number"]) for _ in range(2)]
        self.assertListEqual(expected_seq_start, seq_start)

        expected_seq_end = ["9", "seventy"]
        seq_end = [next(it_dict["snips/number"]) for _ in range(2)]
        self.assertListEqual(expected_seq_end, sorted(seq_end))

    def test_generate_utterance(self):
        # Given
        context = {
            "data": [
                {
                    "text": "this is ",
                },
                {
                    "text": "entity 11",
                    "entity": "entity1",
                    "slot_name": "slot1"
                },
                {
                    "text": " right "
                },
                {
                    "text": "entity 2",
                    "entity": "entity2",
                    "slot_name": "slot1"
                }
            ]
        }
        context_iterator = (context for _ in range(1))

        entities_iterators = {
            "entity1": ("entity one" for _ in range(1)),
            "entity2": ("entity two" for _ in range(1)),
        }

        # When
        utterance = generate_utterance(context_iterator, entities_iterators)

        # Then
        expected_utterance = {
            "data": [
                {
                    "text": "this is ",
                },
                {
                    "text": "entity one ",
                    "entity": "entity1",
                    "slot_name": "slot1"
                },
                {
                    "text": "right "
                },
                {
                    "text": "entity two ",
                    "entity": "entity2",
                    "slot_name": "slot1"
                }
            ]
        }
        self.assertEqual(expected_utterance, utterance)

    @patch("snips_nlu.data_augmentation.get_stop_words")
    def test_capitalize(self, mocked_get_stop_words):
        # Given
        language = LANGUAGE_EN

        def mock_get_stop_words(lang):
            if lang == LANGUAGE_EN:
                return {"the", "and", "you"}
            return {}

        mocked_get_stop_words.side_effect = mock_get_stop_words

        texts = [
            ("the new yorker", "the New Yorker"),
            ("JOHN AND SMITH", "John and Smith"),
            ("you and me", "you and Me")
        ]

        # When
        capitalized_texts = [capitalize(text[0], language) for text in texts]

        # Then
        expected_capitalized_texts = [text[1] for text in texts]
        self.assertSequenceEqual(capitalized_texts, expected_capitalized_texts)

    @patch("snips_nlu.data_augmentation.get_stop_words")
    def test_should_capitalize_only_right_entities(
            self, mocked_get_stop_words):
        # Given
        language = LANGUAGE_EN

        def mock_get_stop_words(lang):
            if lang == LANGUAGE_EN:
                return {"the", "and", "you"}
            return {}

        mocked_get_stop_words.side_effect = mock_get_stop_words
        ratio = 1
        entities = {
            "person": {
                "capitalize": False
            },
            "magazine": {
                "capitalize": True
            }
        }
        utterances = [
            {
                "data": [
                    {
                        "text": "i love "
                    },
                    {
                        "text": "the new yorker",
                        "entity": "magazine"
                    },
                    {
                        "text": " and "
                    },
                    {
                        "text": "rock and rolla",
                        "entity": "magazine"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "let's visit"
                    },
                    {
                        "text": "andrew and smith",
                        "entity": "person"
                    },
                    {
                        "text": " right now"
                    }
                ]
            }
        ]
        random_state = np.random.RandomState(1)

        # When
        capitalized_utterances = capitalize_utterances(
            utterances, entities, language, ratio, random_state)

        # Then
        expected_utterances = [
            {
                "data": [
                    {
                        "text": "i love "
                    },
                    {
                        "text": "the New Yorker",
                        "entity": "magazine"
                    },
                    {
                        "text": " and "
                    },
                    {
                        "text": "Rock and Rolla",
                        "entity": "magazine"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "let's visit"
                    },
                    {
                        "text": "andrew and smith",
                        "entity": "person"
                    },
                    {
                        "text": " right now"
                    }
                ]
            }
        ]
        self.assertEqual(capitalized_utterances, expected_utterances)
