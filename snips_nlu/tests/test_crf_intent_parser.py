import unittest

from snips_nlu.intent_parser.crf_intent_parser import (
    utterance_to_bilou_sample, utterance_to_bio_sample, tags_to_slots_with_bio,
    tags_to_slots_with_bilou)
from snips_nlu.slot_filler.crf_utils import (
    OUTSIDE, BEGINNING_PREFIX, LAST_PREFIX, UNIT_PREFIX, INSIDE_PREFIX)


class TestCRFIntentParser(unittest.TestCase):
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
            expected_tokens = " ".join(c["text"] for c in query).split()
            # When
            sample = utterance_to_bilou_sample(query)

            # Then
            self.assertEqual(expected_tokens, sample["tokens"])
            self.assertEqual(expected_tags, sample["labels"])

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
            expected_tokens = " ".join(c["text"] for c in query).split()
            # When
            sample = utterance_to_bio_sample(query)

            # Then
            self.assertEqual(expected_tokens, sample["tokens"])
            self.assertEqual(expected_tags, sample["labels"])

    def test_tags_to_slots_with_bio(self):
        # Given
        slot_name = "animal"
        tags = [
            {
                "tokens": ["nothing", "here"],
                "tags": [OUTSIDE, OUTSIDE],
                "expected_slots": []
            },
            {
                "tokens": ["i", "am", "a", "blue", "bird"],
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (3, 5),
                        "value": "blue bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["i", "am", "a", "bird"],
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (3, 4),
                        "value": "bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["bird"],
                "tags": [BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 1),
                        "value": "bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["blue", "bird"],
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 2),
                        "value": "blue bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["light", "blue", "bird", "blue", "bird"],
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 3),
                        "value": "light blue bird",
                        "slot_name": slot_name
                    },
                    {
                        "range": (3, 5),
                        "value": "blue bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["bird", "birdy"],
                "tags": [BEGINNING_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 1),
                        "value": "bird",
                        "slot_name": slot_name
                    },
                    {
                        "range": (1, 2),
                        "value": "birdy",
                        "slot_name": slot_name
                    },
                ]
            }

        ]

        for data in tags:
            # When
            slots = tags_to_slots_with_bio(data["tokens"], data["tags"])
            # Then
            self.assertEqual(slots, data["expected_slots"])

    def test_tags_to_slots_with_bilou(self):
        # Given
        slot_name = "animal"
        tags = [
            {
                "tokens": ["nothing", "here"],
                "tags": [OUTSIDE, OUTSIDE],
                "expected_slots": []
            },
            {
                "tokens": ["i", "am", "a", "blue", "bird"],
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         BEGINNING_PREFIX + slot_name,
                         LAST_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (3, 5),
                        "value": "blue bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["i", "am", "a", "bird"],
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         UNIT_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (3, 4),
                        "value": "bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["bird"],
                "tags": [UNIT_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 1),
                        "value": "bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["blue", "bird"],
                "tags": [BEGINNING_PREFIX + slot_name,
                         LAST_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 2),
                        "value": "blue bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["light", "blue", "bird", "blue", "bird"],
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         LAST_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name,
                         LAST_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 3),
                        "value": "light blue bird",
                        "slot_name": slot_name
                    },
                    {
                        "range": (3, 5),
                        "value": "blue bird",
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "tokens": ["bird", "birdy"],
                "tags": [UNIT_PREFIX + slot_name,
                         UNIT_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 1),
                        "value": "bird",
                        "slot_name": slot_name
                    },
                    {
                        "range": (1, 2),
                        "value": "birdy",
                        "slot_name": slot_name
                    },
                ]
            }

        ]

        for data in tags:
            # When
            slots = tags_to_slots_with_bilou(data["tokens"], data["tags"])
            # Then
            self.assertEqual(slots, data["expected_slots"])
