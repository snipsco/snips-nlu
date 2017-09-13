from __future__ import unicode_literals

import unittest

from mock import MagicMock, patch, call

from snips_nlu.builtin_entities import BuiltInEntity
from snips_nlu.constants import MATCH_RANGE, VALUE, ENTITY
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.probabilistic_intent_parser import (
    augment_slots, spans_to_tokens_indexes, ProbabilisticIntentParser,
    DataAugmentationConfig, capitalize, capitalize_utterances,
    generate_slots_permutations)
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import (BEGINNING_PREFIX, INSIDE_PREFIX,
                                             TaggingScheme)
from snips_nlu.tests.utils import BEVERAGE_DATASET
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

            if _tags == tags_1:
                return 0.6
            elif _tags == tags_2:
                return 0.8
            elif _tags == tags_3:
                return 0.2
            elif _tags == tags_4:
                return 0.2
            elif _tags == tags_5:
                return 0.99
            elif _tags == tags_6:
                return 0.0
            elif _tags == tags_7:
                return 0.0
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
            ParsedSlot(value='after 8pm', match_range=(33, 42),
                       entity='snips/datetime', slot_name='end_date')
        ]
        self.assertListEqual(augmented_slots, expected_slots)

    def test_should_fit_only_selected_intents(self):
        # Given
        intents = {"MakeTea"}
        language = Language.EN
        intent_classifier = SnipsIntentClassifier(language)
        tagging_scheme = TaggingScheme.BIO
        mock_coffee_tagger = MagicMock()
        mock_coffee_tagger.tagging_scheme = tagging_scheme
        mock_coffee_tagger.language = language
        mock_tea_tagger = MagicMock()
        mock_tea_tagger.tagging_scheme = tagging_scheme
        mock_tea_tagger.language = language
        taggers = {
            "MakeCoffee": mock_coffee_tagger,
            "MakeTea": mock_tea_tagger
        }
        slot_name_to_entity_mapping = {
            "beverage_temperature": "Temperature",
            "number_of_cups": "snips/number"
        }
        parser = ProbabilisticIntentParser(
            language=language, intent_classifier=intent_classifier,
            crf_taggers=taggers,
            slot_name_to_entity_mapping=slot_name_to_entity_mapping)

        # When
        parser.fit(BEVERAGE_DATASET, intents)

        # Then
        self.assertFalse(mock_coffee_tagger.fit.called)
        self.assertTrue(mock_tea_tagger.fit.called)

    @patch('snips_nlu.slot_filler.crf_tagger.CRFTagger.fit')
    @patch('snips_nlu.slot_filler.crf_tagger.CRFTagger.to_dict')
    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.SnipsIntentClassifier.to_dict')
    def test_should_be_serializable(self, mock_classifier_to_dict,
                                    mock_tagger_to_dict, mock_tagger_fit):
        # Given
        language = Language.EN
        mock_tagger_to_dict.return_value = {
            "mocked_tagger_key": "mocked_tagger_value"}
        mock_classifier_to_dict.return_value = {
            "mocked_dict_key": "mocked_dict_value"}
        intent_classifier = SnipsIntentClassifier(language)
        slot_name_to_entity_mapping = {
            "number_of_cups": "snips/number",
            "beverage_temperature": "Temperature"
        }

        features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2},
                "offsets": [-1, 0]
            }
        ]

        tagging_scheme = TaggingScheme.BIO

        make_coffee_crf = default_crf_model()
        make_tea_crf = default_crf_model()
        make_coffee_tagger = CRFTagger(make_coffee_crf, features_signatures,
                                       tagging_scheme, language)
        make_tea_tagger = CRFTagger(make_tea_crf, features_signatures,
                                    tagging_scheme, language)
        taggers = {
            "MakeCoffee": make_coffee_tagger,
            "MakeTea": make_tea_tagger,
        }

        mock_tagger_fit.side_effect = [make_coffee_tagger, make_tea_tagger]

        parser = ProbabilisticIntentParser(language, intent_classifier,
                                           taggers,
                                           slot_name_to_entity_mapping, None)
        parser.fit(BEVERAGE_DATASET)

        # When
        actual_parser_dict = parser.to_dict()

        # Then
        expected_parser_dict = {
            "data_augmentation_config": {
                "max_utterances": 200,
                "noise_prob": 0.0,
                "min_noise_size": 0,
                "max_noise_size": 0
            },
            "intent_classifier": {
                "mocked_dict_key": "mocked_dict_value"
            },
            "slot_name_to_entity_mapping": {
                "beverage_temperature": "Temperature",
                "number_of_cups": "snips/number"
            },
            "language_code": "en",
            "taggers": {
                "MakeCoffee": {"mocked_tagger_key": "mocked_tagger_value"},
                "MakeTea": {"mocked_tagger_key": "mocked_tagger_value"}
            }
        }
        self.assertDictEqual(actual_parser_dict, expected_parser_dict)

    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.SnipsIntentClassifier.from_dict')
    @patch('snips_nlu.intent_parser.probabilistic_intent_parser'
           '.CRFTagger')
    def test_should_be_deserializable(self, mock_tagger,
                                      mock_classifier_from_dict):
        # When
        language = Language.EN
        mocked_tagger = MagicMock()
        mock_tagger.from_dict.return_value = mocked_tagger
        mocked_tagger.language = language
        parser_dict = {
            "data_augmentation_config": {
                "max_utterances": 200,
                "noise_prob": 0.0,
                "min_noise_size": 0,
                "max_noise_size": 0
            },
            "intent_classifier": {
                "mocked_dict_key": "mocked_dict_value"
            },
            "slot_name_to_entity_mapping": {
                "beverage_temperature": "Temperature",
                "number_of_cups": "snips/number"
            },
            "language_code": "en",
            "taggers": {
                "MakeCoffee": {"mocked_tagger_key1": "mocked_tagger_value1"},
                "MakeTea": {"mocked_tagger_key2": "mocked_tagger_value2"}
            }
        }

        # When
        parser = ProbabilisticIntentParser.from_dict(parser_dict)

        # Then
        mock_classifier_from_dict.assert_called_once_with(
            {"mocked_dict_key": "mocked_dict_value"})
        calls = [call({"mocked_tagger_key1": "mocked_tagger_value1"}),
                 call({"mocked_tagger_key2": "mocked_tagger_value2"})]
        mock_tagger.from_dict.assert_has_calls(calls, any_order=True)

        expected_slot_name_to_entity_mapping = {
            "beverage_temperature": "Temperature",
            "number_of_cups": "snips/number"
        }

        expected_data_augmentation_config = DataAugmentationConfig.from_dict(
            {
                "max_utterances": 200,
                "noise_prob": 0.0,
                "min_noise_size": 0,
                "max_noise_size": 0
            }
        )

        self.assertEqual(parser.language, language)
        self.assertEqual(parser.slot_name_to_entity_mapping,
                         expected_slot_name_to_entity_mapping)
        self.assertEqual(parser.data_augmentation_config,
                         expected_data_augmentation_config)
        self.assertIsNotNone(parser.intent_classifier)
        self.assertItemsEqual(parser.crf_taggers.keys(),
                              ["MakeCoffee", "MakeTea"])

    def test_capitalize(self):
        # Given
        language = Language.EN
        texts = [
            ("university of new york", "University of New York"),
            ("JOHN'S SMITH", "John s Smith"),
            ("is that it", "is that it")
        ]

        # When
        capitalized_texts = [capitalize(text[0], language) for text in texts]

        # Then
        expected_capitalized_texts = [text[1] for text in texts]
        self.assertSequenceEqual(capitalized_texts, expected_capitalized_texts)

    def test_should_capitalize_only_right_entities(self):
        # Given
        language = Language.EN
        ratio = 1
        capitalization_threshold = .1
        utterances = [
            {
                "data": [
                    {
                        "text": "let's go the "
                    },
                    {
                        "text": "university of new york",
                        "entity": "university"
                    },
                    {
                        "text": " right now or "
                    },
                    {
                        "text": "university of London",
                        "entity": "university"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "let's go the "
                    },
                    {
                        "text": "john's smith house",
                        "entity": "someOneHouse"
                    },
                    {
                        "text": " right now"
                    }
                ]
            }
        ]

        # When
        capitalized_utterances = capitalize_utterances(
            utterances, language, ratio, capitalization_threshold)

        # Then
        expected_utterances = [
            {
                "data": [
                    {
                        "text": "let's go the "
                    },
                    {
                        "text": "University of New York",
                        "entity": "university"
                    },
                    {
                        "text": " right now or "
                    },
                    {
                        "text": "University of London",
                        "entity": "university"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "let's go the "
                    },
                    {
                        "text": "john's smith house",
                        "entity": "someOneHouse"
                    },
                    {
                        "text": " right now"
                    }
                ]
            }
        ]
        self.assertEqual(capitalized_utterances, expected_utterances)

    def test_generate_slots_permutations(self):
        # Given
        possible_slots = ["slot1", "slot22"]
        configs = [
            {
                "n_builtins_in_sentence": 0,
                "slots": []
            },
            {
                "n_builtins_in_sentence": 1,
                "slots": [
                    ("slot1",),
                    ("slot22",),
                    ("O",)
                ]
            },
            {
                "n_builtins_in_sentence": 2,
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
                "n_builtins_in_sentence": 3,
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
            slots = generate_slots_permutations(conf["n_builtins_in_sentence"],
                                                possible_slots)
            # Then
            self.assertItemsEqual(conf["slots"], slots)
