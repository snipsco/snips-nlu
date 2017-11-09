from __future__ import unicode_literals

from builtins import next
from builtins import range
import unittest

from mock import patch

from snips_nlu.data_augmentation import (
    get_contexts_iterator, get_entities_iterators,
    generate_utterance)


def np_random_permutation(x):
    return x


class TestDataAugmentation(unittest.TestCase):
    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    def test_context_iterator(self, _):
        # Given
        dataset = {
            "intents": {
                "dummy": {"utterances": list(range(3))}
            }
        }

        # When
        it = get_contexts_iterator(dataset, "dummy")
        context = [next(it) for _ in range(5)]

        # Then
        self.assertEqual(context, [0, 1, 2, 0, 1])

    @patch("numpy.random.permutation", side_effect=np_random_permutation)
    def test_entities_iterators(self, _):
        # Given
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
            }
        }

        # Then
        it_dict = get_entities_iterators(intent_entities)

        # When
        self.assertIn("entity1", it_dict)
        expected_seq = ["entity 1", "entity 11", "entity 111"]
        seq = [next(it_dict["entity1"]) for _ in range(len(expected_seq))]
        self.assertItemsEqual(seq, expected_seq)

        self.assertIn("entity2", it_dict)
        expected_seq = ["entity 2", "entity 22", "entity 222"]
        seq = [next(it_dict["entity2"]) for _ in range(len(expected_seq))]
        self.assertItemsEqual(seq, expected_seq)

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
