# coding=utf-8
from __future__ import unicode_literals

import unittest

from mock import patch, MagicMock

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.config import CRFSlotFillerConfig
from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot
from snips_nlu.slot_filler.crf_slot_filler import CRFSlotFiller, \
    spans_to_tokens_indexes, filter_overlapping_builtins, \
    generate_slots_permutations, exhaustive_slots_permutations
from snips_nlu.slot_filler.crf_utils import TaggingScheme, BEGINNING_PREFIX, \
    INSIDE_PREFIX
from snips_nlu.tests.utils import SAMPLE_DATASET
from snips_nlu.tokenization import tokenize, Token


class TestCRFSlotFiller(unittest.TestCase):
    def test_should_be_serializable_before_fit(self):
        # Given
        language = Language.EN
        features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2, "language_code": language.iso_code},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(tagging_scheme=TaggingScheme.BILOU)

        slot_filler = CRFSlotFiller(features_signatures, config)

        # When
        actual_slot_filler_dict = slot_filler.to_dict()

        # Then
        expected_slot_filler_dict = {
            "crf_model_data": None,
            "features_signatures": [
                {
                    "args": {
                        "n": 1,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language_code": None,
            "config": config.to_dict(),
            "intent": None,
            "slot_name_mapping": None,
        }
        self.assertDictEqual(actual_slot_filler_dict,
                             expected_slot_filler_dict)

    @patch('snips_nlu.slot_filler.crf_slot_filler.deserialize_crf_model')
    def test_should_be_deserializable_before_fit(self,
                                                 mock_deserialize_crf_model):
        # Given
        language = Language.EN
        mock_deserialize_crf_model.return_value = None
        slot_filler_dict = {
            "crf_model_data": None,
            "features_signatures": [
                {
                    "args": {
                        "n": 1,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language_code": None,
            "intent": None,
            "slot_name_mapping": None,
            "config": CRFSlotFillerConfig().to_dict()
        }
        # When
        slot_filler = CRFSlotFiller.from_dict(slot_filler_dict)

        # Then
        expected_features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2, "language_code": language.iso_code},
                "offsets": [-1, 0]
            }
        ]
        expected_language = None
        expected_config = CRFSlotFillerConfig()
        expected_intent = None
        expected_slot_name_mapping = None
        expected_crf_model = None

        self.assertEqual(slot_filler.crf_model, expected_crf_model)
        self.assertListEqual(slot_filler.features_signatures,
                             expected_features_signatures)
        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping,
                         expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(),
                             slot_filler.config.to_dict())

    @patch('snips_nlu.slot_filler.crf_slot_filler.serialize_crf_model')
    def test_should_be_serializable(self, mock_serialize_crf_model):
        language = Language.EN
        # Given
        mock_serialize_crf_model.return_value = "mocked_crf_model_data"
        features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2, "language_code": language.iso_code},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(tagging_scheme=TaggingScheme.BILOU)
        dataset = validate_and_format_dataset(SAMPLE_DATASET)

        slot_filler = CRFSlotFiller(features_signatures, config)
        intent = "dummy_intent_1"
        slot_filler.fit(dataset, intent=intent)

        # When
        actual_slot_filler_dict = slot_filler.to_dict()

        # Then
        expected_slot_filler_dict = {
            "crf_model_data": "mocked_crf_model_data",
            "features_signatures": [
                {
                    "args": {
                        "n": 1,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language_code": "en",
            "config": config.to_dict(),
            "intent": intent,
            "slot_name_mapping": {
                "dummy_intent_1": {
                    "dummy_slot_name": "dummy_entity_1",
                    "dummy_slot_name2": "dummy_entity_2",
                    "dummy_slot_name3": "dummy_entity_2",
                },
                "dummy_intent_2": {
                    "dummy slot nàme": "dummy_entity_1"
                }
            },
        }
        self.assertDictEqual(actual_slot_filler_dict,
                             expected_slot_filler_dict)

    @patch('snips_nlu.slot_filler.crf_slot_filler.deserialize_crf_model')
    def test_should_be_deserializable(self, mock_deserialize_crf_model):
        # Given
        language = Language.EN
        mock_deserialize_crf_model.return_value = None
        slot_filler_dict = {
            "crf_model_data": "mocked_crf_model_data",
            "features_signatures": [
                {
                    "args": {
                        "n": 1,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        0
                    ]
                },
                {
                    "args": {
                        "n": 2,
                        "language_code": language.iso_code
                    },
                    "factory_name": "get_shape_ngram_fn",
                    "offsets": [
                        -1,
                        0
                    ]
                }
            ],
            "language_code": "en",
            "intent": "dummy_intent_1",
            "slot_name_mapping": {
                "dummy_intent_1": {
                    "dummy_slot_name": "dummy_entity_1",
                }
            },
            "config": CRFSlotFillerConfig().to_dict()
        }
        # When
        slot_filler = CRFSlotFiller.from_dict(slot_filler_dict)

        # Then
        mock_deserialize_crf_model.assert_called_once_with(
            "mocked_crf_model_data")
        expected_features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2, "language_code": language.iso_code},
                "offsets": [-1, 0]
            }
        ]
        expected_language = Language.EN
        expected_config = CRFSlotFillerConfig()
        expected_intent = "dummy_intent_1"
        expected_slot_name_mapping = {
            "dummy_intent_1": {
                "dummy_slot_name": "dummy_entity_1",
            }
        }

        self.assertListEqual(slot_filler.features_signatures,
                             expected_features_signatures)
        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping,
                         expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(),
                             slot_filler.config.to_dict())

    def test_should_compute_features(self):
        # Given
        features_signatures = [
            {
                "factory_name": "get_ngram_fn",
                "args": {
                    "n": 1,
                    "use_stemming": False,
                    "language_code": "en",
                    "common_words_gazetteer_name": None
                },
                "offsets": [0]
            },
        ]
        drop_out = {
            "ngram_1": 0.3
        }
        slot_filler_config = CRFSlotFillerConfig(features_drop_out=drop_out,
                                                 random_seed=40)
        slot_filler = CRFSlotFiller(features_signatures, slot_filler_config)

        tokens = tokenize("foo hello world bar", Language.EN)
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        slot_filler.fit(dataset, intent="dummy_intent_1")

        # When
        features_with_drop_out = slot_filler._compute_features(tokens, True)

        # Then
        expected_features = [
            {"ngram_1": "foo"},
            {},
            {"ngram_1": "world"},
            {},
        ]
        self.assertListEqual(expected_features, features_with_drop_out)

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
        language = Language.EN
        text = "Find me a flight before 10pm and after 8pm"
        tokens = tokenize(text, language)
        missing_slots = {"start_date", "end_date"}

        tags = ['O' for _ in tokens]

        def mocked_sequence_probability(_, tags_):
            tags_1 = ['O',
                      'O',
                      'O',
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX,
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX]

            tags_2 = ['O',
                      'O',
                      'O',
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX,
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX]

            tags_3 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

            tags_4 = ['O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX]

            tags_5 = ['O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX]

            tags_6 = ['O',
                      'O',
                      'O',
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX,
                      'O',
                      'O',
                      'O']

            tags_7 = ['O',
                      'O',
                      'O',
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX,
                      'O',
                      'O',
                      'O']

            if tags_ == tags_1:
                return 0.6
            elif tags_ == tags_2:
                return 0.8
            elif tags_ == tags_3:
                return 0.2
            elif tags_ == tags_4:
                return 0.2
            elif tags_ == tags_5:
                return 0.99
            elif tags_ == tags_6:
                return 0.0
            elif tags_ == tags_7:
                return 0.0
            else:
                raise ValueError("Unexpected tag sequence: %s" % tags_)

        slot_filler_config = CRFSlotFillerConfig(
            random_seed=42, exhaustive_permutations_threshold=2)
        slot_filler = CRFSlotFiller(features_signatures=[],
                                    config=slot_filler_config)
        slot_filler.language = Language.EN
        slot_filler.intent = "intent1"
        slot_filler.slot_name_mapping = {
            "intent1": {
                "start_date": "snips/datetime",
                "end_date": "snips/datetime",
            },
            "intent2": {
                "location": "location"
            }
        }

        slot_filler.get_sequence_probability = MagicMock(
            side_effect=mocked_sequence_probability)

        # When
        augmented_slots = slot_filler._augment_slots(text, tokens, tags,
                                                     missing_slots)

        # Then
        expected_slots = [
            ParsedSlot(value='after 8pm', match_range=(33, 42),
                       entity='snips/datetime', slot_name='end_date')
        ]
        self.assertListEqual(augmented_slots, expected_slots)

    def test_filter_overlapping_builtins(self):
        # Given
        language = Language.EN
        text = "Find me a flight before 10pm and after 8pm"
        tokens = tokenize(text, language)
        tags = ['O' for _ in xrange(5)] + ['B-flight'] + ['O' for _ in
                                                          xrange(3)]
        tagging_scheme = TaggingScheme.BIO
        builtin_entities = [
            {
                MATCH_RANGE: (17, 28),
                VALUE: "before 10pm",
                ENTITY: BuiltInEntity.DATETIME
            },
            {
                MATCH_RANGE: (33, 42),
                VALUE: "after 8pm",
                ENTITY: BuiltInEntity.DATETIME
            }
        ]

        # When
        entities = filter_overlapping_builtins(builtin_entities, tokens, tags,
                                               tagging_scheme)

        # Then
        expected_entities = [
            {
                MATCH_RANGE: (33, 42),
                VALUE: "after 8pm",
                ENTITY: BuiltInEntity.DATETIME
            }
        ]
        self.assertEqual(entities, expected_entities)

    def test_exhaustive_slots_permutations(self):
        # Given
        n_builtins = 2
        possible_slots_names = ["a", "b"]

        # When
        perms = exhaustive_slots_permutations(n_builtins, possible_slots_names)

        # Then
        expected_perms = {
            ("a", "a"),
            ("a", "b"),
            ("a", "O"),
            ("b", "b"),
            ("b", "a"),
            ("b", "O"),
            ("O", "a"),
            ("O", "b"),
            ("O", "O"),
        }
        self.assertItemsEqual(perms, expected_perms)

    @patch("snips_nlu.slot_filler.crf_slot_filler"
           ".exhaustive_slots_permutations")
    def test_slot_permutations_should_be_exhaustive(
            self, mocked_exhaustive_slots):
        # Given
        n_builtins = 2
        possible_slots_names = ["a", "b"]
        exhaustive_permutations_threshold = 100

        # When
        generate_slots_permutations(n_builtins, possible_slots_names,
                                    exhaustive_permutations_threshold)

        # Then
        mocked_exhaustive_slots.assert_called_once()

    @patch("snips_nlu.slot_filler.crf_slot_filler"
           ".conservative_slots_permutations")
    def test_slot_permutations_should_be_conservative(
            self, mocked_conservative_slots):
        # Given
        n_builtins = 2
        possible_slots_names = ["a", "b"]
        exhaustive_permutations_threshold = 8

        # When
        generate_slots_permutations(n_builtins, possible_slots_names,
                                    exhaustive_permutations_threshold)

        # Then
        mocked_conservative_slots.assert_called_once()

    def test_generate_slots_permutations(self):
        # Given
        possible_slots = ["slot1", "slot22"]
        configs = [
            {
                "n_builtins_in_sentence": 0,
                "exhaustive_permutations_threshold": 10,
                "slots": []
            },
            {
                "n_builtins_in_sentence": 1,
                "exhaustive_permutations_threshold": 10,
                "slots": [
                    ("slot1",),
                    ("slot22",),
                    ("O",)
                ]
            },
            {
                "n_builtins_in_sentence": 2,
                "exhaustive_permutations_threshold": 4,
                "slots": [
                    ("slot1", "slot22"),
                    ("slot22", "slot1"),
                    ("slot1", "O"),
                    ("O", "slot1"),
                    ("slot22", "O"),
                    ("O", "slot22"),
                    ("O", "O")
                ]
            },
            {
                "n_builtins_in_sentence": 2,
                "exhaustive_permutations_threshold": 100,
                "slots": [
                    ("O", "O"),
                    ("O", "slot1"),
                    ("O", "slot22"),
                    ("slot1", "O"),
                    ("slot1", "slot1"),
                    ("slot1", "slot22"),
                    ("slot22", "O"),
                    ("slot22", "slot1"),
                    ("slot22", "slot22"),
                ]
            },
            {
                "n_builtins_in_sentence": 3,
                "exhaustive_permutations_threshold": 5,
                "slots": [
                    ("slot1", "slot22", "O"),
                    ("slot22", "slot1", "O"),
                    ("slot22", "O", "slot1"),
                    ("slot1", "O", "slot22"),
                    ("O", "slot22", "slot1"),
                    ("O", "slot1", "slot22"),
                    ("O", "O", "slot1"),
                    ("O", "O", "slot22"),
                    ("O", "slot1", "O"),
                    ("O", "slot22", "O"),
                    ("slot1", "O", "O"),
                    ("slot22", "O", "O"),
                    ("O", "O", "O")
                ]
            }
        ]

        for conf in configs:
            # When
            slots = generate_slots_permutations(
                conf["n_builtins_in_sentence"],
                possible_slots,
                conf["exhaustive_permutations_threshold"])
            # Then
            self.assertItemsEqual(conf["slots"], slots)
