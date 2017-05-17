import unittest

from mock import patch

from snips_nlu.result import ParsedSlot
from snips_nlu.slot_filler.crf_utils import (
    OUTSIDE, BEGINNING_PREFIX, LAST_PREFIX, UNIT_PREFIX, INSIDE_PREFIX,
    utterance_to_sample, TaggingScheme, negative_tagging,
    positive_tagging, end_of_bio_slot, start_of_bio_slot, start_of_bilou_slot,
    end_of_bilou_slot, tags_to_slots)
from snips_nlu.tokenization import tokenize, Token


class TestCRFUtils(unittest.TestCase):
    def test_io_tags_to_slots(self):
        # Given
        slot_name = "animal"
        intent_slots_mapping = {"animal": "animal"}
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
                    ParsedSlot(
                        match_range=(7, 16),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "i am a bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(7, 11),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "bird",
                "tags": [INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 4),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "blue bird",
                "tags": [INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 9),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
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
                    ParsedSlot(
                        match_range=(0, 25),
                        value="light blue bird blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "bird birdy",
                "tags": [INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 10),
                        value="bird birdy",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            }

        ]

        for data in tags:
            # When
            slots = tags_to_slots(data["text"], tokenize(data["text"]),
                                  data["tags"], TaggingScheme.IO,
                                  intent_slots_mapping)
            # Then
            self.assertEqual(slots, data["expected_slots"])

    def test_bio_tags_to_slots(self):
        # Given
        slot_name = "animal"
        intent_slots_mapping = {"animal": "animal"}
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
                    ParsedSlot(
                        match_range=(7, 16),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "i am a bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(7, 11),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "bird",
                "tags": [BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 4),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "blue bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 9),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
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
                    ParsedSlot(
                        match_range=(0, 15),
                        value="light blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(16, 25),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "bird birdy",
                "tags": [BEGINNING_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 4),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(5, 10),
                        value="birdy",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "blue bird and white bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         OUTSIDE,
                         INSIDE_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 9),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(14, 24),
                        value="white bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            }
        ]

        for data in tags:
            # When
            slots = tags_to_slots(data["text"], tokenize(data["text"]),
                                  data["tags"], TaggingScheme.BIO,
                                  intent_slots_mapping)
            # Then
            self.assertEqual(slots, data["expected_slots"])

    def test_bilou_tags_to_slots(self):
        # Given
        slot_name = "animal"
        intent_slots_mapping = {"animal": "animal"}
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
                    ParsedSlot(
                        match_range=(7, 16),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "i am a bird",
                "tags": [OUTSIDE, OUTSIDE, OUTSIDE,
                         UNIT_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(7, 11),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "bird",
                "tags": [UNIT_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 4),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "blue bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         LAST_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 9),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
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
                    ParsedSlot(
                        match_range=(0, 15),
                        value="light blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(16, 25),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "bird birdy",
                "tags": [UNIT_PREFIX + slot_name,
                         UNIT_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 4),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(5, 10),
                        value="birdy",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "light bird bird blue bird",
                "tags": [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name,
                         UNIT_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 10),
                        value="light bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(11, 15),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(16, 25),
                        value="blue bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
            {
                "text": "bird bird bird",
                "tags": [LAST_PREFIX + slot_name,
                         BEGINNING_PREFIX + slot_name,
                         UNIT_PREFIX + slot_name],
                "expected_slots": [
                    ParsedSlot(
                        match_range=(0, 4),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(5, 9),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    ),
                    ParsedSlot(
                        match_range=(10, 14),
                        value="bird",
                        entity=slot_name,
                        slot_name=slot_name
                    )
                ]
            },
        ]

        for data in tags:
            # When
            slots = tags_to_slots(data["text"], tokenize(data["text"]),
                                  data["tags"], TaggingScheme.BILOU,
                                  intent_slots_mapping)
            # Then
            self.assertEqual(slots, data["expected_slots"])

    def test_positive_tagging_should_handle_zero_length(self):
        # Given
        slot_name = "animal"
        slot_size = 0

        # When
        tags = []
        for scheme in TaggingScheme:
            tags.append(positive_tagging(scheme, slot_name, slot_size))

        # Then
        expected_tags = [[]] * len(TaggingScheme)
        self.assertEqual(tags, expected_tags)

    @patch('snips_nlu.slot_filler.crf_utils.positive_tagging')
    def test_utterance_to_sample(self, mocked_positive_tagging):
        # Given
        def mock_positive_tagging(tagging_scheme, slot, slot_size):
            return [INSIDE_PREFIX + slot for _ in xrange(slot_size)]

        mocked_positive_tagging.side_effect = mock_positive_tagging
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
        sample = utterance_to_sample(query_data, TaggingScheme.IO)

        # Then
        self.assertEqual(sample, expected_sample)

    @patch('snips_nlu.slot_filler.crf_utils.positive_tagging')
    def test_utterance_to_sample_with_partial_slots(self,
                                                    mocked_positive_tagging):

        # Given
        def mock_positive_tagging(tagging_scheme, slot, slot_size):
            return [INSIDE_PREFIX + slot for _ in xrange(slot_size)]

        mocked_positive_tagging.side_effect = mock_positive_tagging
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
        sample = utterance_to_sample(query_data, TaggingScheme.IO)

        # Then
        mocked_positive_tagging.assert_called()
        self.assertEqual(sample, expected_sample)

    def test_negative_tagging(self):
        # Given
        size = 3

        # When
        tagging = negative_tagging(size)

        # Then
        expected_tagging = [OUTSIDE, OUTSIDE, OUTSIDE]
        self.assertListEqual(tagging, expected_tagging)

    def test_positive_tagging_with_io(self):
        # Given
        tagging_scheme = TaggingScheme.IO
        slot_name = "animal"
        slot_size = 3

        # When
        tags = positive_tagging(tagging_scheme, slot_name, slot_size)

        # Then
        t = INSIDE_PREFIX + slot_name
        expected_tags = [t, t, t]
        self.assertListEqual(tags, expected_tags)

    def test_positive_tagging_with_bio(self):
        # Given
        tagging_scheme = TaggingScheme.BIO
        slot_name = "animal"
        slot_size = 3

        # When
        tags = positive_tagging(tagging_scheme, slot_name, slot_size)

        # Then
        expected_tags = [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name, INSIDE_PREFIX + slot_name]
        self.assertListEqual(tags, expected_tags)

    def test_positive_tagging_with_bilou(self):
        # Given
        tagging_scheme = TaggingScheme.BILOU
        slot_name = "animal"
        slot_size = 3

        # When
        tags = positive_tagging(tagging_scheme, slot_name, slot_size)

        # Then
        expected_tags = [BEGINNING_PREFIX + slot_name,
                         INSIDE_PREFIX + slot_name, LAST_PREFIX + slot_name]
        self.assertListEqual(tags, expected_tags)

    def test_positive_tagging_with_bilou_unit(self):
        # Given
        tagging_scheme = TaggingScheme.BILOU
        slot_name = "animal"
        slot_size = 1

        # When
        tags = positive_tagging(tagging_scheme, slot_name, slot_size)

        # Then
        expected_tags = [UNIT_PREFIX + slot_name]
        self.assertListEqual(tags, expected_tags)

    def test_start_of_bio_slot(self):
        # Given
        tags = [
            OUTSIDE,
            BEGINNING_PREFIX,
            INSIDE_PREFIX,
            OUTSIDE,
            INSIDE_PREFIX,
            OUTSIDE,
            BEGINNING_PREFIX,
            OUTSIDE,
            INSIDE_PREFIX,
            BEGINNING_PREFIX,
            OUTSIDE,
            BEGINNING_PREFIX,
            BEGINNING_PREFIX,
            INSIDE_PREFIX,
            INSIDE_PREFIX
        ]

        # When
        starts_of_bio = [start_of_bio_slot(tags, i) for i in range(len(tags))]

        # Then
        expected_starts = [
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            False
        ]

        self.assertListEqual(starts_of_bio, expected_starts)

    def test_end_of_bio_slot(self):
        # Given
        tags = [
            OUTSIDE,
            BEGINNING_PREFIX,
            INSIDE_PREFIX,
            OUTSIDE,
            INSIDE_PREFIX,
            OUTSIDE,
            BEGINNING_PREFIX,
            OUTSIDE,
            INSIDE_PREFIX,
            BEGINNING_PREFIX,
            OUTSIDE,
            BEGINNING_PREFIX,
            BEGINNING_PREFIX,
            INSIDE_PREFIX,
            INSIDE_PREFIX
        ]

        # When
        ends_of_bio = [end_of_bio_slot(tags, i) for i in range(len(tags))]

        # Then
        expected_ends = [
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            False,
            True
        ]

        self.assertListEqual(ends_of_bio, expected_ends)

    def test_start_of_bilou_slot(self):
        # Given
        tags = [
            OUTSIDE,
            LAST_PREFIX,
            UNIT_PREFIX,
            BEGINNING_PREFIX,
            UNIT_PREFIX,
            INSIDE_PREFIX,
            LAST_PREFIX,
            LAST_PREFIX,
            UNIT_PREFIX,
            UNIT_PREFIX,
            LAST_PREFIX,
            OUTSIDE,
            LAST_PREFIX,
            BEGINNING_PREFIX,
            INSIDE_PREFIX,
            INSIDE_PREFIX,
            LAST_PREFIX
        ]

        # When
        starts_of_bilou = [start_of_bilou_slot(tags, i) for i in
                           range(len(tags))]

        # Then
        expected_starts = [
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False
        ]

        self.assertListEqual(starts_of_bilou, expected_starts)

    def test_end_of_bilou_slot(self):
        # Given
        tags = [
            OUTSIDE,
            LAST_PREFIX,
            UNIT_PREFIX,
            BEGINNING_PREFIX,
            UNIT_PREFIX,
            INSIDE_PREFIX,
            LAST_PREFIX,
            LAST_PREFIX,
            UNIT_PREFIX,
            UNIT_PREFIX,
            LAST_PREFIX,
            OUTSIDE,
            INSIDE_PREFIX,
            BEGINNING_PREFIX,
            OUTSIDE,
            BEGINNING_PREFIX,
            INSIDE_PREFIX,
            INSIDE_PREFIX,
            LAST_PREFIX
        ]

        # When
        ends_of_bilou = [end_of_bilou_slot(tags, i) for i in range(len(tags))]

        # Then
        expected_ends = [
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            True
        ]

        self.assertListEqual(ends_of_bilou, expected_ends)
