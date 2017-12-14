# coding=utf-8
from __future__ import unicode_literals

import io
import json
import os
import traceback as tb
import unittest
from copy import deepcopy

from mock import Mock, patch

import snips_nlu
import snips_nlu.version
from snips_nlu.configs.intent_parser import ProbabilisticIntentParserConfig
from snips_nlu.configs.nlu_engine import NLUEngineConfig
from snips_nlu.configs.slot_filler import CRFSlotFillerConfig
from snips_nlu.constants import DATA, TEXT, LANGUAGE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.nlu_engine import SnipsNLUEngine, enrich_slots
from snips_nlu.result import Result, ParsedSlot, IntentClassificationResult
from snips_nlu.tests.utils import (SAMPLE_DATASET, get_empty_dataset,
                                   TEST_PATH,
                                   BEVERAGE_DATASET)


class TestSnipsNLUEngine(unittest.TestCase):
    def test_should_use_parsers_sequentially(self):
        # Given
        language = Language.EN

        input_text = "hello world"

        mocked_parser1 = Mock()
        intent_result1 = None
        intent_entities1 = []
        mocked_parser1.get_intent.return_value = intent_result1
        mocked_parser1.get_entities.return_value = intent_entities1

        mocked_parser2 = Mock()
        intent_result2 = IntentClassificationResult(
            intent_name='mocked_intent2', probability=0.7)
        intent_entities2_empty = []
        intent_entities2 = [
            ParsedSlot(match_range=(3, 5), value='mocked_value',
                       entity='mocked_entity', slot_name='mocked_slot_name')]
        mocked_parser2.get_intent.return_value = intent_result2

        def mock_get_slots(text, intent):
            assert text == input_text
            if intent == intent_result2.intent_name:
                return intent_entities2
            return intent_entities2_empty

        mocked_parser2.get_slots = Mock(side_effect=mock_get_slots)
        mocked_entities = {
            "mocked_entity": {
                "automatically_extensible": True,
                "utterances": dict()
            }
        }
        engine = SnipsNLUEngine()
        engine.language = language
        engine.entities = mocked_entities
        engine.deterministic_parser = mocked_parser1
        engine.probabilistic_parser = mocked_parser2

        # When
        parse = engine.parse(input_text)

        # Then
        expected_parse = Result(input_text, intent_result2, intent_entities2)
        self.assertDictEqual(parse, expected_parse.as_dict())

    def test_should_handle_empty_dataset(self):
        # Given
        dataset = validate_and_format_dataset(get_empty_dataset(Language.EN))
        engine = SnipsNLUEngine().fit(dataset)

        # When
        result = engine.parse("hello world")

        # Then
        self.assertEqual(result, Result("hello world", None, None).as_dict())

    @patch('snips_nlu.nlu_engine.ProbabilisticIntentParser.to_dict')
    @patch('snips_nlu.nlu_engine.DeterministicIntentParser.to_dict')
    def test_should_be_serializable(self, mock_deterministic_parser_to_dict,
                                    mock_probabilistic_parser_to_dict):
        # Given
        mocked_deterministic_parser_dict = {
            "mocked_deterministic_parser_key":
                "mocked_deterministic_parser_value"
        }
        mock_deterministic_parser_to_dict.return_value = \
            mocked_deterministic_parser_dict
        mocked_proba_parser_dict = {
            "mocked_proba_parser_key": "mocked_proba_parser_value"}
        mock_probabilistic_parser_to_dict.return_value = \
            mocked_proba_parser_dict
        engine = SnipsNLUEngine().fit(BEVERAGE_DATASET)

        # When
        actual_engine_dict = engine.to_dict()

        # Then
        expected_engine_dict = {
            "entities": {
                "Temperature": {
                    "automatically_extensible": True,
                    "utterances": {
                        "boiling": "hot",
                        "cold": "cold",
                        "hot": "hot",
                        "iced": "cold"
                    }
                }
            },
            "config": NLUEngineConfig().to_dict(),
            "model": {
                "deterministic_parser": mocked_deterministic_parser_dict,
                "probabilistic_parser": mocked_proba_parser_dict
            },
            "model_version": snips_nlu.version.__model_version__,
            "training_package_version": snips_nlu.__version__
        }

        self.assertDictEqual(actual_engine_dict, expected_engine_dict)

    @patch('snips_nlu.nlu_engine.ProbabilisticIntentParser.from_dict')
    @patch('snips_nlu.nlu_engine.DeterministicIntentParser.from_dict')
    def test_should_be_deserializable(self,
                                      mock_deterministic_parser_from_dict,
                                      mock_probabilistic_parser_from_dict):
        # When
        mocked_deterministic_parser_dict = {
            "mocked_deterministic_parser_key":
                "mocked_deterministic_parser_value"
        }
        mocked_proba_parser_dict = {
            "mocked_proba_based_parser_key": "mocked_proba_parser_value"
        }
        entities = {
            "Temperature": {
                "automatically_extensible": True,
                "utterances": {
                    "boiling": "hot",
                    "cold": "cold",
                    "hot": "hot",
                    "iced": "cold"
                }
            }
        }
        engine_dict = {
            "entities": entities,
            "config": NLUEngineConfig(),
            "model": {
                "deterministic_parser": mocked_deterministic_parser_dict,
                "probabilistic_parser": mocked_proba_parser_dict
            },
            "model_version": snips_nlu.version.__model_version__,
            "training_package_version": snips_nlu.__version__
        }
        engine = SnipsNLUEngine.from_dict(engine_dict)

        # Then
        mock_deterministic_parser_from_dict.assert_called_once_with(
            mocked_deterministic_parser_dict)

        mock_probabilistic_parser_from_dict.assert_called_once_with(
            mocked_proba_parser_dict)

        self.assertDictEqual(engine.entities, entities)
        self.assertDictEqual(engine.config.to_dict(),
                             NLUEngineConfig().to_dict())

    def test_should_parse_after_deserialization(self):
        # Given
        dataset = BEVERAGE_DATASET
        engine = SnipsNLUEngine().fit(dataset)
        text = "Give me 3 cups of hot tea please"

        # When
        engine_dict = engine.to_dict()
        deserialized_engine = SnipsNLUEngine.from_dict(engine_dict)
        result = deserialized_engine.parse(text)

        # Then
        try:
            json.dumps(engine_dict).encode("utf-8")
        except Exception, e:  # pylint: disable=W0703
            trace = tb.format_exc()
            self.fail("SnipsNLUEngine dict should be json serializable "
                      "to utf-8.\n{}\n{}".format(e, trace))
        expected_slots = [
            ParsedSlot((8, 9), '3', 'snips/number',
                       'number_of_cups').as_dict(),
            ParsedSlot((18, 21), 'hot', 'Temperature',
                       'beverage_temperature').as_dict()
        ]
        self.assertEqual(result['text'], text)
        self.assertEqual(result['intent']['intent_name'], 'MakeTea')
        self.assertListEqual(result['slots'], expected_slots)

    def test_should_fail_when_missing_intents(self):
        # Given
        incomplete_intents = {"MakeCoffee"}
        engine = SnipsNLUEngine()

        # Then
        with self.assertRaises(Exception) as context:
            engine.fit(BEVERAGE_DATASET, intents=incomplete_intents)

        self.assertTrue("These intents must be trained: set([u'MakeTea'])"
                        in context.exception)

    def test_should_use_fitted_tagger(self):
        # Given
        text = "Give me 3 cups of hot tea please"
        fitted_slot_filler = SnipsNLUEngine().get_fitted_slot_filler(
            BEVERAGE_DATASET, "MakeTea")

        # When
        engine = SnipsNLUEngine()
        engine.add_fitted_slot_filler("MakeTea", fitted_slot_filler.to_dict())
        engine.fit(BEVERAGE_DATASET, intents=["MakeCoffee"])
        result = engine.parse(text)

        # Then
        expected_slots = [
            ParsedSlot((8, 9), '3', 'snips/number',
                       'number_of_cups').as_dict(),
            ParsedSlot((18, 21), 'hot', 'Temperature',
                       'beverage_temperature').as_dict()
        ]
        self.assertEqual(result['text'], text)
        self.assertEqual(result['intent']['intent_name'], 'MakeTea')
        self.assertListEqual(result['slots'], expected_slots)

    def test_should_be_serializable_after_fitted_tagger_is_added(self):
        # Given
        text = "Give me 3 cups of hot tea please"
        engine = SnipsNLUEngine()
        trained_slot_filler_coffee = engine.get_fitted_slot_filler(
            BEVERAGE_DATASET, "MakeCoffee")
        trained_slot_filler_tea = engine.get_fitted_slot_filler(
            BEVERAGE_DATASET, "MakeTea")

        # When
        engine = SnipsNLUEngine()
        engine.add_fitted_slot_filler("MakeCoffee",
                                      trained_slot_filler_coffee.to_dict())
        engine.add_fitted_slot_filler("MakeTea",
                                      trained_slot_filler_tea.to_dict())
        engine.fit(BEVERAGE_DATASET, intents=[])

        try:
            engine_dict = engine.to_dict()
            new_engine = SnipsNLUEngine.from_dict(engine_dict)
        except Exception, e:  # pylint: disable=W0703
            self.fail('Exception raised: %s\n%s' %
                      (e.message, tb.format_exc()))
        result = new_engine.parse(text)

        # Then
        expected_slots = [
            ParsedSlot((8, 9), '3', 'snips/number',
                       'number_of_cups').as_dict(),
            ParsedSlot((18, 21), 'hot', 'Temperature',
                       'beverage_temperature').as_dict()
        ]
        self.assertEqual(result['text'], text)
        self.assertEqual(result['intent']['intent_name'], 'MakeTea')
        self.assertListEqual(result['slots'], expected_slots)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_slots")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.deterministic_intent_parser"
           ".DeterministicIntentParser.get_intent")
    def test_should_handle_keyword_entities(self, mocked_regex_get_intent,
                                            mocked_crf_get_intent,
                                            mocked_crf_get_slots,
                                            mocked_default_features):
        # Given
        language = Language.EN
        dataset = {
            "snips_nlu_version": "1.1.1",
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy_1",
                                    "entity": "dummy_entity_1",
                                    "slot_name": "dummy_slot_name"
                                },
                                {
                                    "text": " dummy_2",
                                    "entity": "dummy_entity_2",
                                    "slot_name": "other_dummy_slot_name"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "dummy_entity_1": {
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "data": [
                        {
                            "value": "dummy1",
                            "synonyms": [
                                "dummy1",
                                "dummy1_bis"
                            ]
                        },
                        {
                            "value": "dummy2",
                            "synonyms": [
                                "dummy2",
                                "dummy2_bis"
                            ]
                        }
                    ]
                },
                "dummy_entity_2": {
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "data": [
                        {
                            "value": "dummy2",
                            "synonyms": [
                                "dummy2"
                            ]
                        }
                    ]
                }
            },
            "language": language.iso_code
        }

        mocked_default_features.return_value = []
        mocked_crf_intent = IntentClassificationResult("dummy_intent_1", 1.0)
        mocked_crf_slots = [ParsedSlot(match_range=(0, 7),
                                       value="dummy_3",
                                       entity="dummy_entity_1",
                                       slot_name="dummy_slot_name"),
                            ParsedSlot(match_range=(8, 15),
                                       value="dummy_4",
                                       entity="dummy_entity_2",
                                       slot_name="other_dummy_slot_name")]

        mocked_regex_get_intent.return_value = None
        mocked_crf_get_intent.return_value = mocked_crf_intent
        mocked_crf_get_slots.return_value = mocked_crf_slots

        engine = SnipsNLUEngine()
        text = "dummy_3 dummy_4"

        # When
        engine = engine.fit(dataset)
        result = engine.parse(text)

        # Then
        expected_result = Result(
            text, parsed_intent=mocked_crf_intent,
            parsed_slots=[ParsedSlot(match_range=(8, 15), value="dummy_4",
                                     entity="dummy_entity_2",
                                     slot_name="other_dummy_slot_name")]) \
            .as_dict()
        self.assertEqual(result, expected_result)

    @patch("snips_nlu.slot_filler.feature_functions.default_features")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_slots")
    @patch(
        "snips_nlu.intent_parser.probabilistic_intent_parser"
        ".ProbabilisticIntentParser.get_intent")
    @patch("snips_nlu.intent_parser.deterministic_intent_parser"
           ".DeterministicIntentParser.get_intent")
    def test_synonyms_should_point_to_base_value(self, mocked_deter_get_intent,
                                                 mocked_crf_get_intent,
                                                 mocked_crf_get_slots,
                                                 mocked_default_features):
        # Given
        language = Language.EN
        dataset = {
            "snips_nlu_version": "1.1.1",
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "dummy_1",
                                    "entity": "dummy_entity_1",
                                    "slot_name": "dummy_slot_name"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "dummy_entity_1": {
                    "use_synonyms": True,
                    "automatically_extensible": False,
                    "data": [
                        {
                            "value": "dummy1",
                            "synonyms": [
                                "dummy1",
                                "dummy1_bis"
                            ]
                        }
                    ]
                }
            },
            "language": language.iso_code
        }

        mocked_default_features.return_value = []
        mocked_crf_intent = IntentClassificationResult("dummy_intent_1", 1.0)
        mocked_crf_slots = [ParsedSlot(match_range=(0, 10), value="dummy1_bis",
                                       entity="dummy_entity_1",
                                       slot_name="dummy_slot_name")]

        mocked_deter_get_intent.return_value = None
        mocked_crf_get_intent.return_value = mocked_crf_intent
        mocked_crf_get_slots.return_value = mocked_crf_slots

        engine = SnipsNLUEngine().fit(dataset)
        text = "dummy1_bis"

        # When
        result = engine.parse(text)

        # Then
        expected_result = Result(
            text, parsed_intent=mocked_crf_intent,
            parsed_slots=[ParsedSlot(match_range=(0, 10), value="dummy1",
                                     entity="dummy_entity_1",
                                     slot_name="dummy_slot_name")])
        self.assertEqual(result, expected_result.as_dict())

    def test_enrich_slots(self):
        # Given
        slots = [
            # Adjacent
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((2, 6), "", "", ""),
                    ParsedSlot((8, 10), "", "", "")
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", ""),
                    ParsedSlot((2, 6), "", "", ""),
                    ParsedSlot((8, 10), "", "", "")
                ]
            },
            # Equality
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((6, 8), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Inclusion
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((5, 7), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Cross upper
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((7, 10), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Cross lower
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((5, 7), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            },
            # Full overlap
            {
                "slots": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ],
                "other_slots": [
                    ParsedSlot((4, 12), "", "", ""),
                ],
                "enriched": [
                    ParsedSlot((0, 2), "", "", ""),
                    ParsedSlot((6, 8), "", "", "")
                ]
            }
        ]

        for data in slots:
            # When
            enriched = enrich_slots(data["slots"], data["other_slots"])

            # Then
            self.assertEqual(enriched, data["enriched"])

    def test_should_parse_naughty_strings(self):
        # Given
        dataset = SAMPLE_DATASET
        naughty_strings_path = os.path.join(TEST_PATH, "resources",
                                            "naughty_strings.txt")
        with io.open(naughty_strings_path, encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        # When
        engine = SnipsNLUEngine().fit(dataset)

        # Then
        for s in naughty_strings:
            try:
                engine.parse(s)
            except:  # pylint: disable=W0702
                trace = tb.format_exc()
                self.fail('Exception raised:\n %s' % trace)

    def test_should_fit_with_naughty_strings(self):
        # Given
        naughty_strings_path = os.path.join(TEST_PATH, "resources",
                                            "naughty_strings.txt")
        with io.open(naughty_strings_path, encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]
        utterances = [{DATA: [{TEXT: naughty_string}]} for naughty_string in
                      naughty_strings]

        # When
        naughty_dataset = validate_and_format_dataset({
            "intents": {
                "naughty_intent": {
                    "utterances": utterances
                }
            },
            "entities": dict(),
            "language": "en",
            "snips_nlu_version": "0.0.1"
        })

        # Then
        try:
            SnipsNLUEngine().fit(naughty_dataset)
        except:  # pylint: disable=W0702
            trace = tb.format_exc()
            self.fail('Exception raised:\n %s' % trace)

    def test_engine_should_fit_with_builtins_entities(self):
        # Given
        language = Language.EN
        dataset = validate_and_format_dataset({
            "intents": {
                "dummy": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "10p.m.",
                                    "entity": "snips/datetime",
                                    "slot_name": "startTime"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {
                "snips/datetime": {}
            },
            "language": language.iso_code,
            "snips_nlu_version": "0.0.1"
        })

        # When / Then
        try:
            SnipsNLUEngine().fit(dataset)
        except:  # pylint: disable=W0702
            self.fail("NLU engine should fit builtin")

    def test_get_fitted_slot_filler_should_return_same_slot_filler_as_fit(
            self):
        # Given
        intent = "MakeCoffee"
        slot_filler_config = CRFSlotFillerConfig(random_seed=42)
        config = NLUEngineConfig(ProbabilisticIntentParserConfig(
            crf_slot_filler_config=slot_filler_config))
        trained_engine = SnipsNLUEngine(config).fit(BEVERAGE_DATASET)

        # When
        engine = SnipsNLUEngine(config)
        slot_filler = engine.get_fitted_slot_filler(BEVERAGE_DATASET, intent)

        # Then
        expected_slot_filler = \
            trained_engine.probabilistic_parser.slot_fillers[intent]
        self.assertEqual(slot_filler.crf_model.state_features_,
                         expected_slot_filler.crf_model.state_features_)
        self.assertEqual(slot_filler.crf_model.transition_features_,
                         expected_slot_filler.crf_model.transition_features_)

    @patch("snips_nlu.intent_parser.probabilistic_intent_parser."
           "ProbabilisticIntentParser.get_slots")
    @patch("snips_nlu.intent_parser.probabilistic_intent_parser."
           "ProbabilisticIntentParser.get_intent")
    def test_parse_should_call_probabilistic_intent_parser_when_given_intent(
            self, mocked_probabilistic_get_intent,
            mocked_probabilistic_get_slots):
        # Given
        dataset = deepcopy(SAMPLE_DATASET)
        dataset["entities"]["dummy_entity_1"][
            "automatically_extensible"] = True
        engine = SnipsNLUEngine().fit(dataset)
        intent = "dummy_intent_1"
        text = "This is another weird weird query"

        intent_classif_result = IntentClassificationResult(intent, .8)
        expected_intent_classif_result = IntentClassificationResult(intent,
                                                                    1.0)
        mocked_probabilistic_get_intent.return_value = intent_classif_result

        parsed_slots = [ParsedSlot(match_range=(16, 27), value="weird weird",
                                   entity="dummy_entity_1",
                                   slot_name="dummy slot nàme")]
        mocked_probabilistic_get_slots.return_value = parsed_slots

        # When
        parse = engine.parse(text, intent=intent)

        # Then
        mocked_probabilistic_get_intent.assert_called_once()
        mocked_probabilistic_get_slots.assert_called_once()
        expected_parse = Result(text, expected_intent_classif_result,
                                parsed_slots).as_dict()
        self.assertEqual(parse, expected_parse)

    def test_nlu_engine_should_train_and_parse_in_all_languages(self):
        # Given
        text = "brew me an expresso"
        for l in Language:
            dataset = deepcopy(BEVERAGE_DATASET)
            dataset[LANGUAGE] = l.iso_code
            engine = SnipsNLUEngine()

            # When / Then
            try:
                engine = engine.fit(dataset)
            except:  # pylint: disable=W0702
                self.fail("Could not fit engine in '%s': %s"
                          % (l.iso_code, tb.format_exc()))

            try:
                engine.parse(text)
            except:  # pylint: disable=W0702
                self.fail("Could not fit engine in '%s': %s"
                          % (l.iso_code, tb.format_exc()))
