import unittest

from mock import mock

from snips_nlu.slot_filler.crf_utils import (
    OUTSIDE, BEGINNING_PREFIX, LAST_PREFIX, UNIT_PREFIX, INSIDE_PREFIX,
    io_tags_to_slots, utterance_to_sample, Tagging)
from snips_nlu.slot_filler.crf_utils import (bio_tags_to_slots,
                                             bilou_tags_to_slots)
from snips_nlu.tokenization import tokenize, Token


class TestCRFUtils(unittest.TestCase):
    def test_io_tags_to_slots(self):
        # Given
        slot_name = "animal"
        tags = [
            {
                "text": "",
                "tags": [],
                "expected_slots": []
            },
            {
                "text": "nothing here",
                "tags": [OUTSIDE, OUTSIDE],
                "expected_slots": []
            },
            {
                "text": "i am a blue bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (7, 16),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "i am a bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (7, 11),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "bird",
                "tags": [INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 4),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "blue bird",
                "tags": [INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 9),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "light blue bird blue bird",
                "tags": [INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 25),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "bird birdy",
                "tags": [INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 10),
                        "slot_name": slot_name
                    }
                ]
            }

        ]

        for data in tags:
            # When
            slots = io_tags_to_slots(data["tags"], tokenize(data["text"]))
            # Then
            self.assertEqual(slots, data["expected_slots"])

    def test_boi_tags_to_slots(self):
        # Given
        slot_name = "animal"
        tags = [
            {
                "text": "",
                "tags": [],
                "expected_slots": []
            },
            {
                "text": "nothing here",
                "tags": [OUTSIDE, OUTSIDE],
                "expected_slots": []
            },
            {
                "text": "i am a blue bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (7, 16),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "i am a bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (7, 11),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "bird",
                "tags": [BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 4),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "blue bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 9),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "light blue bird blue bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 15),
                        "slot_name": slot_name
                    },
                    {
                        "range": (16, 25),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "bird birdy",
                "tags": [BEGINNING_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 4),
                        "slot_name": slot_name
                    },
                    {
                        "range": (5, 10),
                        "slot_name": slot_name
                    }
                ]
            }

        ]

        for data in tags:
            # When
            slots = bio_tags_to_slots(data["tags"], tokenize(data["text"]))
            # Then
            self.assertEqual(slots, data["expected_slots"])

    def test_bilou_tags_to_slots(self):
        # Given
        slot_name = "animal"
        tags = [
            {
                "text": "",
                "tags": [],
                "expected_slots": []
            },
            {
                "text": "nothing here",
                "tags": [OUTSIDE, OUTSIDE],
                "expected_slots": []
            },
            {
                "text": "i am a blue bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         BEGINNING_PREFIX + slot_name,
                         LAST_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (7, 16),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "i am a bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         UNIT_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (7, 11),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "bird",
                "tags": [UNIT_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 4),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "blue bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         LAST_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 9),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "light blue bird blue bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         LAST_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name,
                         LAST_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 15),
                        "slot_name": slot_name
                    },
                    {
                        "range": (16, 25),
                        "slot_name": slot_name
                    }
                ]
            },
            {
                "text": "bird birdy",
                "tags": [UNIT_PREFIX + slot_name,
                         UNIT_PREFIX + slot_name],
                "expected_slots": [
                    {
                        "range": (0, 4),
                        "slot_name": slot_name
                    },
                    {
                        "range": (5, 10),
                        "slot_name": slot_name
                    }
                ]
            }

        ]

        for data in tags:
            # When
            slots = bilou_tags_to_slots(data["tags"], tokenize(data["text"]))
            # Then
            self.assertEqual(slots, data["expected_slots"])

    def test_utterance_to_io_sample(self):
        # Given
        slot_name = "animal"
        queries = [
            ([
                 {
                     "text": "nothing here",
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
                     "text": " blue bird",
                     "slot_name": slot_name
                 }
             ],
             [INSIDE_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "i am a b",

                 },
                 {
                     "text": "lue bird",
                     "slot_name": slot_name
                 },
                 {
                     "text": " hey",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, OUTSIDE, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name, OUTSIDE])
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
                     "text": "nothing here",
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
                     "text": " blue bird",
                     "slot_name": slot_name
                 }
             ],
             [BEGINNING_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              LAST_PREFIX + slot_name, BEGINNING_PREFIX + slot_name,
              LAST_PREFIX + slot_name]),
            ([
                 {
                     "text": "i am a b",

                 },
                 {
                     "text": "lue bird",
                     "slot_name": slot_name
                 },
                 {
                     "text": " hey",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, OUTSIDE, BEGINNING_PREFIX + slot_name,
              LAST_PREFIX + slot_name, OUTSIDE])
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
                     "text": "nothing here",
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
                     "text": " blue bird",
                     "slot_name": slot_name
                 }
             ],
             [BEGINNING_PREFIX + slot_name, INSIDE_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name, BEGINNING_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name]),
            ([
                 {
                     "text": "i am a b",

                 },
                 {
                     "text": "lue bird",
                     "slot_name": slot_name
                 },
                 {
                     "text": " hey",
                     "slot_name": slot_name
                 }
             ],
             [OUTSIDE, OUTSIDE, OUTSIDE, OUTSIDE, BEGINNING_PREFIX + slot_name,
              INSIDE_PREFIX + slot_name, OUTSIDE])
        ]

        for query, expected_tags in queries:
            expected_tokens = tokenize("".join(c["text"] for c in query))
            # When
            sample = utterance_to_sample(query, Tagging.BIO)

            # Then
            self.assertEqual(expected_tokens, sample["tokens"])
            self.assertEqual(expected_tags, sample["tags"])

    @mock.patch("snips_nlu.slot_filler.crf_utils")
    def test_utterance_to_sample(self, mock_crf_utils):
        # Given
        def mocked_positive_tagging(tagging, slot_name, slot_size):
            return [INSIDE_PREFIX + slot_name for _ in xrange(slot_size)]

        mock_crf_utils.positive_tagging = mock.Mock(
            side_effect=mocked_positive_tagging)
        slot_name = "animal"
        query_data = [{"text": "i am a "},
                      {"text": "beautiful bird", "slot_name": slot_name}]
        expected_tagging = [OUTSIDE, OUTSIDE, OUTSIDE,
                            INSIDE_PREFIX + slot_name,
                            INSIDE_PREFIX + slot_name]
        expected_tokens = [Token(value='i', start=0, end=1),
                           Token(value='am', start=2, end=4),
                           Token(value='a', start=5, end=6),
                           Token(value='beautiful', start=7, end=16),
                           Token(value='bird', start=17, end=21)]
        expected_sample = {"tokens": expected_tokens,
                           "tags": expected_tagging}

        # When
        sample = utterance_to_sample(query_data, Tagging.IO)

        # Then
        self.assertEqual(sample, expected_sample)

    @mock.patch("snips_nlu.slot_filler.crf_utils")
    def test_utterance_to_sample_with_partial_slots(self, mock_crf_utils):
        # Given
        def mocked_positive_tagging(tagging, slot_name, slot_size):
            return [INSIDE_PREFIX + slot_name for _ in xrange(slot_size)]

        mock_crf_utils.positive_tagging = mock.Mock(
            side_effect=mocked_positive_tagging)
        slot_name = "animal"
        query_data = [{"text": "i am a b"},
                      {"text": "eautiful bird", "slot_name": slot_name}]
        expected_tagging = [OUTSIDE, OUTSIDE, OUTSIDE, OUTSIDE,
                            INSIDE_PREFIX + slot_name,
                            INSIDE_PREFIX + slot_name]

        expected_tokens = [Token(value='i', start=0, end=1),
                           Token(value='am', start=2, end=4),
                           Token(value='a', start=5, end=6),
                           Token(value='b', start=7, end=8),
                           Token(value='eautiful', start=8, end=16),
                           Token(value='bird', start=17, end=21)]

        expected_sample = {"tokens": expected_tokens, "tags": expected_tagging}

        # When
        sample = utterance_to_sample(query_data, Tagging.IO)

        # Then
        self.assertEqual(sample, expected_sample)
