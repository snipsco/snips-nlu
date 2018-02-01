# coding=utf-8
from __future__ import unicode_literals

import io
import os
import traceback as tb
import unittest
from builtins import range

from mock import patch, MagicMock

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.constants import (
    RES_MATCH_RANGE, VALUE, ENTITY, DATA, TEXT, SLOT_NAME)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.pipeline.configs.slot_filler import CRFSlotFillerConfig
from snips_nlu.result import _slot
from snips_nlu.slot_filler.crf_slot_filler import (
    CRFSlotFiller, _spans_to_tokens_indexes, _filter_overlapping_builtins,
    _generate_slots_permutations, _exhaustive_slots_permutations)
from snips_nlu.slot_filler.crf_utils import (
    TaggingScheme, BEGINNING_PREFIX, INSIDE_PREFIX)
from snips_nlu.slot_filler.feature_factory import (
    IsDigitFactory, ShapeNgramFactory, NgramFactory)
from snips_nlu.tests.utils import SAMPLE_DATASET, BEVERAGE_DATASET, TEST_PATH
from snips_nlu.tokenization import tokenize, Token


class TestCRFSlotFiller(unittest.TestCase):
    def test_should_get_slots(self):
        # Given
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        config = CRFSlotFillerConfig(random_seed=42)
        intent = "MakeTea"
        slot_filler = CRFSlotFiller(config)
        slot_filler.fit(dataset, intent)

        # When
        slots = slot_filler.get_slots("make me two cups of tea")

        # Then
        expected_slots = [
            _slot(match_range=(8, 11),
                  value='two',
                  entity='snips/number',
                  slot_name='number_of_cups')]
        self.assertListEqual(slots, expected_slots)

    def test_should_parse_naughty_strings(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        naughty_strings_path = os.path.join(TEST_PATH, "resources",
                                            "naughty_strings.txt")
        with io.open(naughty_strings_path, encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        # When
        slot_filler = CRFSlotFiller().fit(dataset, "dummy_intent_1")

        # Then
        for s in naughty_strings:
            try:
                slot_filler.get_slots(s)
            except:  # pylint: disable=W0702
                trace = tb.format_exc()
                self.fail('Exception raised:\n %s' % trace)

    def test_should_fit_with_naughty_strings_no_tags(self):
        # Given
        naughty_strings_path = os.path.join(TEST_PATH, "resources",
                                            "naughty_strings.txt")
        with io.open(naughty_strings_path, encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        utterances = [{DATA: [{TEXT: naughty_string}]} for naughty_string in
                      naughty_strings]

        # When
        naughty_dataset = {
            "intents": {
                "naughty_intent": {
                    "utterances": utterances
                }
            },
            "entities": dict(),
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }

        # Then
        try:
            CRFSlotFiller().fit(naughty_dataset, "naughty_intent")
        except:  # pylint: disable=W0702
            trace = tb.format_exc()
            self.fail('Exception raised:\n %s' % trace)

    @unittest.skip("Skip until the bug is fixed on the Rust side")
    def test_should_fit_and_parse_with_non_ascii_tags(self):
        # Given
        inputs = ("string%s" % i for i in range(10))
        utterances = [{
            DATA: [{
                TEXT: string,
                ENTITY: "non_ascìi_entïty",
                SLOT_NAME: "non_ascìi_slöt"
            }]
        } for string in inputs]

        # When
        naughty_dataset = {
            "intents": {
                "naughty_intent": {
                    "utterances": utterances
                }
            },
            "entities": {
                "non_ascìi_entïty": {
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "data": []
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }

        naughty_dataset = validate_and_format_dataset(naughty_dataset)

        # Then
        try:
            slot_filler = CRFSlotFiller()
            slot_filler.fit(naughty_dataset, "naughty_intent")
            slots = slot_filler.get_slots("string0")
            expected_slot = {
                'entity': 'non_ascìi_entïty',
                'range': [0, 7],
                'slotName': u'non_ascìi_slöt',
                'value': u'string0'
            }
            self.assertListEqual([expected_slot], slots)
        except:  # pylint: disable=W0702
            trace = tb.format_exc()
            self.fail('Exception raised:\n %s' % trace)

    def test_should_get_slots_after_deserialization(self):
        # Given
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        config = CRFSlotFillerConfig(random_seed=42)
        intent = "MakeTea"
        slot_filler = CRFSlotFiller(config)
        slot_filler.fit(dataset, intent)
        deserialized_slot_filler = CRFSlotFiller.from_dict(
            slot_filler.to_dict())

        # When
        slots = deserialized_slot_filler.get_slots("make me two cups of tea")

        # Then
        expected_slots = [
            _slot(match_range=(8, 11),
                  value='two',
                  entity='snips/number',
                  slot_name='number_of_cups')]
        self.assertListEqual(slots, expected_slots)

    def test_should_be_serializable_before_fit(self):
        # Given
        features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(
            tagging_scheme=TaggingScheme.BILOU,
            feature_factory_configs=features_factories)

        slot_filler = CRFSlotFiller(config)

        # When
        actual_slot_filler_dict = slot_filler.to_dict()

        # Then
        expected_slot_filler_dict = {
            "unit_name": "crf_slot_filler",
            "crf_model_data": None,
            "language_code": None,
            "config": config.to_dict(),
            "intent": None,
            "slot_name_mapping": None,
        }
        self.assertDictEqual(actual_slot_filler_dict,
                             expected_slot_filler_dict)

    @patch('snips_nlu.slot_filler.crf_slot_filler._deserialize_crf_model')
    def test_should_be_deserializable_before_fit(self,
                                                 mock_deserialize_crf_model):
        # Given
        mock_deserialize_crf_model.return_value = None
        features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        slot_filler_config = CRFSlotFillerConfig(
            feature_factory_configs=features_factories)
        slot_filler_dict = {
            "unit_name": "crf_slot_filler",
            "crf_model_data": None,
            "language_code": None,
            "intent": None,
            "slot_name_mapping": None,
            "config": slot_filler_config.to_dict()
        }

        # When
        slot_filler = CRFSlotFiller.from_dict(slot_filler_dict)

        # Then
        expected_features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        expected_language = None
        expected_config = CRFSlotFillerConfig(
            feature_factory_configs=expected_features_factories)
        expected_intent = None
        expected_slot_name_mapping = None
        expected_crf_model = None

        self.assertEqual(slot_filler.crf_model, expected_crf_model)
        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping,
                         expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(),
                             slot_filler.config.to_dict())

    @patch('snips_nlu.slot_filler.crf_slot_filler._serialize_crf_model')
    def test_should_be_serializable(self, mock_serialize_crf_model):
        # Given
        mock_serialize_crf_model.return_value = "mocked_crf_model_data"
        features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(
            tagging_scheme=TaggingScheme.BILOU,
            feature_factory_configs=features_factories)
        dataset = validate_and_format_dataset(SAMPLE_DATASET)

        slot_filler = CRFSlotFiller(config)
        intent = "dummy_intent_1"
        slot_filler.fit(dataset, intent=intent)

        # When
        actual_slot_filler_dict = slot_filler.to_dict()

        # Then
        expected_feature_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1, "language_code": "en"},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        expected_config = CRFSlotFillerConfig(
            tagging_scheme=TaggingScheme.BILOU,
            feature_factory_configs=expected_feature_factories)
        expected_slot_filler_dict = {
            "unit_name": "crf_slot_filler",
            "crf_model_data": "mocked_crf_model_data",
            "language_code": "en",
            "config": expected_config.to_dict(),
            "intent": intent,
            "slot_name_mapping": {
                "dummy_slot_name": "dummy_entity_1",
                "dummy_slot_name2": "dummy_entity_2",
                "dummy_slot_name3": "dummy_entity_2",
            }
        }
        self.assertDictEqual(actual_slot_filler_dict,
                             expected_slot_filler_dict)

    @patch('snips_nlu.slot_filler.crf_slot_filler._deserialize_crf_model')
    def test_should_be_deserializable(self, mock_deserialize_crf_model):
        # Given
        language = Language.EN
        mock_deserialize_crf_model.return_value = None
        feature_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        slot_filler_config = CRFSlotFillerConfig(
            feature_factory_configs=feature_factories)
        slot_filler_dict = {
            "unit_name": "crf_slot_filler",
            "crf_model_data": "mocked_crf_model_data",
            "language_code": "en",
            "intent": "dummy_intent_1",
            "slot_name_mapping": {
                "dummy_intent_1": {
                    "dummy_slot_name": "dummy_entity_1",
                }
            },
            "config": slot_filler_config.to_dict()
        }
        # When
        slot_filler = CRFSlotFiller.from_dict(slot_filler_dict)

        # Then
        mock_deserialize_crf_model.assert_called_once_with(
            "mocked_crf_model_data")
        expected_language = Language.EN
        expected_feature_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1, "language_code": language.iso_code},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        expected_config = CRFSlotFillerConfig(
            feature_factory_configs=expected_feature_factories)
        expected_intent = "dummy_intent_1"
        expected_slot_name_mapping = {
            "dummy_intent_1": {
                "dummy_slot_name": "dummy_entity_1",
            }
        }

        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping,
                         expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(),
                             slot_filler.config.to_dict())

    def test_should_compute_features(self):
        # Given
        features_factories = [
            {
                "factory_name": NgramFactory.name,
                "args": {
                    "n": 1,
                    "use_stemming": False,
                    "common_words_gazetteer_name": None
                },
                "offsets": [0],
                "drop_out": 0.3
            },
        ]
        slot_filler_config = CRFSlotFillerConfig(
            feature_factory_configs=features_factories, random_seed=40)
        slot_filler = CRFSlotFiller(slot_filler_config)

        tokens = tokenize("foo hello world bar", Language.EN)
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        slot_filler.fit(dataset, intent="dummy_intent_1")

        # When
        features_with_drop_out = slot_filler.compute_features(tokens, True)

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
        indexes = _spans_to_tokens_indexes(spans, tokens)

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
        slot_filler = CRFSlotFiller(config=slot_filler_config)
        slot_filler.language = Language.EN
        slot_filler.intent = "intent1"
        slot_filler.slot_name_mapping = {
            "start_date": "snips/datetime",
            "end_date": "snips/datetime",
        }

        slot_filler._get_sequence_probability = MagicMock(
            side_effect=mocked_sequence_probability)

        slot_filler.compute_features = MagicMock(return_value=None)

        # When
        # pylint: disable=protected-access
        augmented_slots = slot_filler._augment_slots(text, tokens, tags,
                                                     missing_slots)
        # pylint: enable=protected-access

        # Then
        expected_slots = [
            _slot(value='after 8pm', match_range=(33, 42),
                  entity='snips/datetime', slot_name='end_date')
        ]
        self.assertListEqual(augmented_slots, expected_slots)

    def test_filter_overlapping_builtins(self):
        # Given
        language = Language.EN
        text = "Find me a flight before 10pm and after 8pm"
        tokens = tokenize(text, language)
        tags = ['O' for _ in range(5)] + ['B-flight'] + ['O' for _ in range(3)]
        tagging_scheme = TaggingScheme.BIO
        builtin_entities = [
            {
                RES_MATCH_RANGE: (17, 28),
                VALUE: "before 10pm",
                ENTITY: BuiltInEntity.DATETIME
            },
            {
                RES_MATCH_RANGE: (33, 42),
                VALUE: "after 8pm",
                ENTITY: BuiltInEntity.DATETIME
            }
        ]

        # When
        entities = _filter_overlapping_builtins(builtin_entities, tokens, tags,
                                                tagging_scheme)

        # Then
        expected_entities = [
            {
                RES_MATCH_RANGE: (33, 42),
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
        perms = _exhaustive_slots_permutations(n_builtins,
                                               possible_slots_names)

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
        self.assertSetEqual(set(perms), expected_perms)

    @patch("snips_nlu.slot_filler.crf_slot_filler"
           "._exhaustive_slots_permutations")
    def test_slot_permutations_should_be_exhaustive(
            self, mocked_exhaustive_slots):
        # Given
        n_builtins = 2
        possible_slots_names = ["a", "b"]
        exhaustive_permutations_threshold = 100

        # When
        _generate_slots_permutations(n_builtins, possible_slots_names,
                                     exhaustive_permutations_threshold)

        # Then
        mocked_exhaustive_slots.assert_called_once()

    @patch("snips_nlu.slot_filler.crf_slot_filler"
           "._conservative_slots_permutations")
    def test_slot_permutations_should_be_conservative(
            self, mocked_conservative_slots):
        # Given
        n_builtins = 2
        possible_slots_names = ["a", "b"]
        exhaustive_permutations_threshold = 8

        # When
        _generate_slots_permutations(n_builtins, possible_slots_names,
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
            slots = _generate_slots_permutations(
                conf["n_builtins_in_sentence"],
                possible_slots,
                conf["exhaustive_permutations_threshold"])
            # Then
            self.assertSetEqual(set(conf["slots"]), set(slots))
