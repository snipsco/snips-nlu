import unittest

from snips_nlu.slot_filler.crf_utils import (
    OUTSIDE, BEGINNING_PREFIX, LAST_PREFIX, UNIT_PREFIX, INSIDE_PREFIX,
    io_tags_to_slots)
from snips_nlu.slot_filler.crf_utils import (bio_tags_to_slots,
                                             bilou_tags_to_slots)
from snips_nlu.tokenization import tokenize


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
