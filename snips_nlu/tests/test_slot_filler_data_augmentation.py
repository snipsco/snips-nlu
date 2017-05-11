from __future__ import unicode_literals

import unittest

from mock import patch

from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.slot_filler.data_augmentation import (
    get_contexts_iterator, get_intent_entities, get_entities_iterators,
    generate_utterance, get_noise_iterator)


def np_random_permutation(x):
    return x


class TestDataAugmentation(unittest.TestCase):
    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    def test_context_iterator(self, _):
        # Given
        seq = range(3)

        # When
        it = get_contexts_iterator(seq)
        context = [next(it) for _ in xrange(5)]

        # Then
        self.assertEqual(context, [0, 1, 2, 0, 1])

    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    def test_entities_iterators(self, _):
        # Given
        dataset = {
            "snips_nlu_version": "1.1.1",
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "entity 1",
                                    "entity": "entity1",
                                    "slot_name": "slot1"
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "entity 2",
                                    "entity": "entity2",
                                    "slot_name": "slot1"
                                },
                                {
                                    "text": " right"
                                }
                            ]
                        }
                    ],
                    "engineType": "regex"
                },
                "intent2": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "this is ",
                                },
                                {
                                    "text": "entity 2",
                                    "entity": "entity2",
                                    "slot_name": "slot2"
                                }
                            ]
                        }
                    ],
                    "engineType": "regex"
                }
            },
            "entities": {
                "entity1": {
                    "data": [
                        {
                            "value": "entity 1",
                            "synonyms": ["entity 1", "entity 11", "entity 111"]
                        },
                        {
                            "value": "entity 3",
                            "synonyms": ["entity 3", "entity 33"]
                        }
                    ],
                    "use_synonyms": True,
                    "automatically_extensible": False
                },
                "entity2": {
                    "data": [
                        {
                            "value": "entity 2",
                            "synonyms": ["entity 2", "entity 22"]
                        },
                        {
                            "value": "entity 44",
                            "synonyms": ["entity 44", "entity 444444"]
                        }
                    ],
                    "use_synonyms": False,
                    "automatically_extensible": False
                }
            },
            "language": "en"
        }
        dataset = validate_and_format_dataset(dataset)

        intent_name = "intent1"
        intent_entities = get_intent_entities(dataset, intent_name)

        # Then
        it_dict = get_entities_iterators(dataset, intent_entities)

        # When
        self.assertIn("entity1", it_dict)
        expected_seq = ["entity 1", "entity 11", "entity 111", "entity 3",
                        "entity 33", "entity 1", "entity 11"]
        seq = [next(it_dict["entity1"]) for _ in xrange(len(expected_seq))]
        self.assertEqual(seq, expected_seq)

        self.assertIn("entity2", it_dict)
        expected_seq = ["entity 2", "entity 44", "entity 2"]
        seq = [next(it_dict["entity2"]) for _ in xrange(len(expected_seq))]
        self.assertEqual(seq, expected_seq)

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
        context_iterator = (context for _ in xrange(1))

        entities_iterators = {
            "entity1": ("entity one" for _ in xrange(1)),
            "entity2": ("entity two" for _ in xrange(1)),
        }

        # When
        utterance = generate_utterance(context_iterator, entities_iterators,
                                       noise_iterator=(_ for _ in xrange(0)),
                                       noise_prob=0)

        # Then
        expected_utterance = {
            "data": [
                {
                    "text": "this is ",
                },
                {
                    "text": "entity one",
                    "entity": "entity1",
                    "slot_name": "slot1"
                },
                {
                    "text": " right "
                },
                {
                    "text": "entity two",
                    "entity": "entity2",
                    "slot_name": "slot1"
                }
            ]
        }
        self.assertEqual(utterance, expected_utterance)

    @patch("random.random")
    def test_generate_utterance_with_noise(self, mocked_random):
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
        context_iterator = (context for _ in xrange(1))

        entities_iterators = {
            "entity1": ("entity one" for _ in xrange(1)),
            "entity2": ("entity two" for _ in xrange(1)),
        }
        noise_iterator = (a for a in ("hi", "hello", "how", "are", "you"))

        global SEED
        SEED = 0

        def random():
            global SEED
            SEED += 1
            return 0.99 if SEED % 2 == 0 else 0.

        mocked_random.side_effect = random

        # When
        utterance = generate_utterance(context_iterator, entities_iterators,
                                       noise_iterator=noise_iterator,
                                       noise_prob=0.5)

        # Then
        expected_utterance = {
            "data": [
                {
                    "text": "this is ",
                },
                {
                    "text": "hi "
                },
                {
                    "text": "entity one",
                    "entity": "entity1",
                    "slot_name": "slot1"
                },
                {
                    "text": " right "
                },

                {
                    "text": "hello "
                },
                {
                    "text": "entity two",
                    "entity": "entity2",
                    "slot_name": "slot1"
                }
            ]
        }

        self.assertEqual(utterance, expected_utterance)

    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    @patch("random.randint")
    @patch("random.choice")
    @patch("snips_nlu.slot_filler.data_augmentation.get_subtitles")
    def test_get_noise_iterator(self, mocked_get_subtitles, mocked_choice,
                                mocked_randint, _):
        # Given
        language = Language.EN
        min_size, max_size = 2, 3

        mocked_subtitles = ["a b c d", "e", "f g h"]
        mocked_get_subtitles.return_value = mocked_subtitles
        mocked_choice.return_value = 2
        mocked_randint.return_value = 0

        it = get_noise_iterator(language, min_size, max_size)

        # When
        seq = [next(it) for _ in xrange(3)]

        # Then
        expected_seq = ["a b", "e f", "a b"]
        self.assertEqual(seq, expected_seq)
