import unittest

from snips_nlu.intent_parser.crf_intent_parser import utterance_to_sample
from snips_nlu.slot_filler.crf_utils import (
    OUTSIDE, BEGINNING_PREFIX, LAST_PREFIX, UNIT_PREFIX, Tagging,
    INSIDE_PREFIX)
from snips_nlu.tokenization import tokenize


class TestCRFIntentParser(unittest.TestCase):
    def test_utterance_to_io_sample(self):
        # Given
        slot_name = "animal"
        queries = [
            ([
                 {
                     "text": "nothing",

                 },
                 {
                     "text": "here",
                 }
             ],
             [OUTSIDE, OUTSIDE]),
            ([
                 {
                     "text": "i am a ",

                 },
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "i am a ",

                 },
                 {
                     "text": "bird",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "bird",
                     "slot_name": slot_name
                 }
             ],
             [INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [INSIDE_PREFIX + slot_name, INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "light blue bird",
                     "slot_name": slot_name
                 },
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [INSIDE_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name])
        ]

        for query, expected_tags in queries:
            expected_tokens = tokenize("".join(c["text"] for c in query))
            # When
            sample = utterance_to_sample(query, Tagging.IO)

            # Then
            self.assertEqual(expected_tokens, sample["tokens"])
            self.assertEqual(expected_tags, sample["tags"])

    def test_utterance_to_bilou_sample(self):
        # Given
        slot_name = "animal"
        queries = [
            ([
                 {
                     "text": "nothing",

                 },
                 {
                     "text": "here",
                 }
             ],
             [OUTSIDE, OUTSIDE]),
            ([
                 {
                     "text": "i am a ",

                 },
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, BEGINNING_PREFIX + slot_name,
              LAST_PREFIX + slot_name]),
            ([
                 {
                     "text": "i am a ",

                 },
                 {
                     "text": "bird",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, UNIT_PREFIX + slot_name]),
            ([
                 {
                     "text": "bird",
                     "slot_name": slot_name
                 }
             ],
             [UNIT_PREFIX + slot_name]),
            ([
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [BEGINNING_PREFIX + slot_name, LAST_PREFIX + slot_name]),
            ([
                 {
                     "text": "light blue bird",
                     "slot_name": slot_name
                 },
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [BEGINNING_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              LAST_PREFIX + slot_name, BEGINNING_PREFIX + slot_name,
              LAST_PREFIX + slot_name])
        ]

        for query, expected_tags in queries:
            expected_tokens = tokenize("".join(c["text"] for c in query))
            # When
            sample = utterance_to_sample(query, Tagging.BILOU)

            # Then
            self.assertEqual(expected_tokens, sample["tokens"])
            self.assertEqual(expected_tags, sample["tags"])

    def test_utterance_to_bio_sample(self):
        # Given
        slot_name = "animal"
        queries = [
            ([
                 {
                     "text": "nothing",

                 },
                 {
                     "text": "here",
                 }
             ],
             [OUTSIDE, OUTSIDE]),
            ([
                 {
                     "text": "i am a ",

                 },
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, BEGINNING_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "i am a ",

                 },
                 {
                     "text": "bird",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, BEGINNING_PREFIX + slot_name]),
            ([
                 {
                     "text": "bird",
                     "slot_name": slot_name
                 }
             ],
             [BEGINNING_PREFIX + slot_name]),
            ([
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [BEGINNING_PREFIX + slot_name, INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "light blue bird",
                     "slot_name": slot_name
                 },
                 {
                     "text": "blue bird",
                     "slot_name": slot_name
                 }
             ],
             [BEGINNING_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name, BEGINNING_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name])
        ]

        for query, expected_tags in queries:
            expected_tokens = tokenize("".join(c["text"] for c in query))
            # When
            sample = utterance_to_sample(query, Tagging.BIO)

            # Then
            self.assertEqual(expected_tokens, sample["tokens"])
            self.assertEqual(expected_tags, sample["tags"])
