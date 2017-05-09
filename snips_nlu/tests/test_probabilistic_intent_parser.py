import unittest

from mock import MagicMock

from snips_nlu.built_in_entities import BuiltInEntity
from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY
from snips_nlu.intent_parser.probabilistic_intent_parser import (
    augment_slots, spans_to_tokens_indexes, ProbabilisticIntentParser)
from snips_nlu.result import ParsedSlot
from snips_nlu.slot_filler.crf_utils import (BEGINNING_PREFIX, INSIDE_PREFIX,
                                             TaggingScheme)
from snips_nlu.tokenization import Token, tokenize


class TestProbabilisticIntentParser(unittest.TestCase):
    def test_spans_to_tokens_indexes(self):
        # Given
        spans = [
            (0, 1),
            (2, 6),
            (5, 6),
            (9, 15)
        ]
        tokens = [
            Token(value="abc", start=0, end=3, stem="abc"),
            Token(value="def", start=4, end=7, stem="def"),
            Token(value="ghi", start=10, end=13, stem="ghi")
        ]

        # When
        indexes = spans_to_tokens_indexes(spans, tokens)

        # Then
        expected_indexes = [[0], [0, 1], [1], [2]]
        self.assertListEqual(indexes, expected_indexes)

    def test_augment_slots(self):
        # Given
        text = "Find me a flight before 10pm and after 8pm"
        tokens = tokenize(text)
        intent_slots_mapping = {
            "start_date": "snips/datetime",
            "end_date": "snips/datetime",
        }
        missing_slots = {"start_date", "end_date"}
        builtin_entities = [
            {
                MATCH_RANGE: (16, 28),
                VALUE: " before 10pm",
                ENTITY: BuiltInEntity.DATETIME
            },
            {
                MATCH_RANGE: (33, 42),
                VALUE: "after 8pm",
                ENTITY: BuiltInEntity.DATETIME
            }
        ]

        tags = ['O' for _ in tokens]

        def mocked_sequence_probability(_tokens, _tags):
            first_tags = ['O' for _ in _tokens]
            first_tags[4] = '%sstart_date' % BEGINNING_PREFIX
            first_tags[5] = '%sstart_date' % INSIDE_PREFIX
            first_tags[7] = '%send_date' % BEGINNING_PREFIX
            first_tags[8] = '%send_date' % INSIDE_PREFIX

            second_tags = ['O' for _ in _tokens]
            second_tags[4] = '%send_date' % BEGINNING_PREFIX
            second_tags[5] = '%send_date' % INSIDE_PREFIX
            second_tags[7] = '%sstart_date' % BEGINNING_PREFIX
            second_tags[8] = '%sstart_date' % INSIDE_PREFIX

            if _tags == first_tags:
                return 0.6
            if _tags == second_tags:
                return 0.8
            else:
                raise ValueError("Unexpected tag sequence: %s" % _tags)

        tagger = MagicMock()
        tagger.get_sequence_probability = MagicMock(
            side_effect=mocked_sequence_probability)
        tagger.tagging_scheme = TaggingScheme.BIO

        # When
        augmented_slots = augment_slots(text, tokens, tags, tagger,
                                        intent_slots_mapping,
                                        builtin_entities, missing_slots)

        # Then
        expected_slots = [
            ParsedSlot(value='before 10pm', match_range=(17, 28),
                       entity='snips/datetime', slot_name='end_date'),
            ParsedSlot(value='after 8pm', match_range=(33, 42),
                       entity='snips/datetime', slot_name='start_date')
        ]
        self.assertListEqual(augmented_slots, expected_slots)
