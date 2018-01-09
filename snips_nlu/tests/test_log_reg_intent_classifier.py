# coding=utf-8
from __future__ import unicode_literals

import unittest

import numpy as np
from mock import patch

from snips_nlu.constants import INTENTS, DATA, UTTERANCES, RES_INTENT_NAME
from snips_nlu.dataset import validate_and_format_dataset, get_text_from_chunks
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.log_reg_classifier import (
    LogRegIntentClassifier)
from snips_nlu.intent_classifier.log_reg_classifier_utils import \
    remove_builtin_slots, get_noise_it, generate_smart_noise, \
    generate_noise_utterances, add_unknown_word_to_utterances, \
    build_training_data
from snips_nlu.languages import Language
from snips_nlu.pipeline.configs.intent_classifier import (
    IntentClassifierConfig, IntentClassifierDataAugmentationConfig)
from snips_nlu.tests.utils import SAMPLE_DATASET, get_empty_dataset, \
    BEVERAGE_DATASET


# pylint: disable=W0613
def get_mocked_augment_utterances(dataset, intent_name, language,
                                  min_utterances, capitalization_ratio,
                                  random_state):
    return dataset[INTENTS][intent_name][UTTERANCES]


# pylint: enable=W0613


class TestLogRegIntentClassifier(unittest.TestCase):
    def test_intent_classifier_should_get_intent(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        classifier = LogRegIntentClassifier().fit(dataset)
        text = "This is a dummy_3 query from another intent"

        # When
        res = classifier.get_intent(text)
        intent = res[RES_INTENT_NAME]

        # Then
        expected_intent = "dummy_intent_2"

        self.assertEqual(intent, expected_intent)

    def test_intent_classifier_should_get_intent_when_filter(self):
        # Given
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        classifier = LogRegIntentClassifier().fit(dataset)

        # When
        text1 = "Make me two cups of tea"
        res1 = classifier.get_intent(text1, ["MakeCoffee", "MakeTea"])

        text2 = "Make me two cups of tea"
        res2 = classifier.get_intent(text2, ["MakeCoffee"])

        text3 = "bla bla bla"
        res3 = classifier.get_intent(text3, ["MakeCoffee"])

        # Then
        self.assertEqual("MakeTea", res1[RES_INTENT_NAME])
        self.assertEqual("MakeCoffee", res2[RES_INTENT_NAME])
        self.assertEqual(None, res3)

    def test_should_get_none_if_empty_dataset(self):
        # Given
        dataset = validate_and_format_dataset(get_empty_dataset(Language.EN))
        classifier = LogRegIntentClassifier().fit(dataset)
        text = "this is a dummy query"

        # When
        intent = classifier.get_intent(text)

        # Then
        expected_intent = None
        self.assertEqual(intent, expected_intent)

    @patch('snips_nlu.intent_classifier.featurizer.Featurizer.to_dict')
    def test_should_be_serializable(self, mock_to_dict):
        # Given
        mocked_dict = {"mocked_featurizer_key": "mocked_featurizer_value"}

        mock_to_dict.return_value = mocked_dict

        dataset = validate_and_format_dataset(SAMPLE_DATASET)

        intent_classifier = LogRegIntentClassifier().fit(dataset)
        coeffs = intent_classifier.classifier.coef_.tolist()
        intercept = intent_classifier.classifier.intercept_.tolist()

        # When
        classifier_dict = intent_classifier.to_dict()

        # Then
        intent_list = SAMPLE_DATASET[INTENTS].keys() + [None]
        expected_dict = {
            "unit_name": "log_reg_intent_classifier",
            "config": IntentClassifierConfig().to_dict(),
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": intent_list,
            "featurizer": mocked_dict
        }
        self.assertEqual(expected_dict, classifier_dict)

    @patch('snips_nlu.intent_classifier.featurizer.Featurizer.from_dict')
    def test_should_be_deserializable(self, mock_from_dict):
        # Given
        mocked_featurizer = Featurizer(Language.EN, None)
        mock_from_dict.return_value = mocked_featurizer

        intent_list = ["MakeCoffee", "MakeTea", None]

        coeffs = [
            [1.23, 4.5],
            [6.7, 8.90],
            [1.01, 2.345],
        ]

        intercept = [
            0.34,
            0.41,
            -0.98
        ]

        config = IntentClassifierConfig().to_dict()

        classifier_dict = {
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": intent_list,
            "config": config,
            "featurizer": mocked_featurizer.to_dict(),
        }

        # When
        classifier = LogRegIntentClassifier.from_dict(classifier_dict)

        # Then
        self.assertEqual(classifier.intent_list, intent_list)
        self.assertIsNotNone(classifier.featurizer)
        self.assertListEqual(classifier.classifier.coef_.tolist(), coeffs)
        self.assertListEqual(classifier.classifier.intercept_.tolist(),
                             intercept)
        self.assertDictEqual(classifier.config.to_dict(), config)

    @patch("snips_nlu.intent_classifier.log_reg_classifier"
           ".build_training_data")
    def test_empty_vocabulary_should_fit_and_return_none_intent(
            self, mocked_build_training):
        # Given
        language = Language.EN
        dataset = {
            "snips_nlu_version": "0.0.1",
            "entities": {
                "dummy_entity_1": {
                    "automatically_extensible": True,
                    "use_synonyms": False,
                    "data": [
                        {
                            "value": "...",
                            "synonyms": [],
                        }
                    ]
                }
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "...",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                }
                            ]
                        }
                    ]
                }
            },
            "language": language.iso_code
        }
        dataset = validate_and_format_dataset(dataset)

        text = " "
        noise_size = 6
        utterance = [text] + [text] * noise_size
        labels = [1] + [None] * noise_size
        intent_list = ["dummy_intent_1", None]
        mocked_build_training.return_value = utterance, labels, intent_list

        # When / Then
        intent_classifier = LogRegIntentClassifier().fit(dataset)
        intent = intent_classifier.get_intent("no intent there")
        self.assertEqual(intent, None)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils"
           ".augment_utterances")
    def test_should_build_training_data_with_no_stemming_no_noise(
            self, mocked_augment_utterances):
        # Given
        dataset = SAMPLE_DATASET
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances
        random_state = np.random.RandomState(1)

        # When
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            noise_factor=0)
        utterances, _, intent_mapping = build_training_data(
            dataset, Language.EN, data_augmentation_config, random_state)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA]) for intent
                               in dataset[INTENTS].values() for utterance in
                               intent[UTTERANCES]]
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1']
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(expected_intent_mapping, intent_mapping)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noises")
    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils"
           ".augment_utterances")
    def test_should_build_training_data_with_noise(
            self, mocked_augment_utterances, mocked_get_noises):
        # Given
        mocked_noises = ["mocked_noise_%s" % i for i in xrange(100)]
        mocked_get_noises.return_value = mocked_noises
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances

        num_intents = 3
        utterances_length = 5
        num_queries_per_intent = 3
        fake_utterance = {
            "data": [
                {"text": " ".join("1" for _ in xrange(utterances_length))}
            ]
        }
        dataset = {
            "intents": {
                unicode(i): {
                    "utterances": [fake_utterance] * num_queries_per_intent
                } for i in xrange(num_intents)
            }
        }
        random_state = np.random.RandomState(1)

        # When
        np.random.seed(42)
        noise_factor = 2
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            noise_factor=noise_factor, unknown_word_prob=0,
            unknown_words_replacement_string=None)
        utterances, _, intent_mapping = build_training_data(
            dataset, Language.EN, data_augmentation_config, random_state)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA])
                               for intent in dataset[INTENTS].values()
                               for utterance in intent[UTTERANCES]]
        np.random.seed(42)
        noise = list(mocked_noises)
        noise_size = int(min(noise_factor * num_queries_per_intent,
                             len(noise)))
        noise_it = get_noise_it(mocked_noises, utterances_length, 0,
                                random_state)
        noisy_utterances = [next(noise_it) for _ in xrange(noise_size)]
        expected_utterances += list(noisy_utterances)
        expected_intent_mapping = dataset["intents"].keys() + [None]
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noises")
    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils"
           ".augment_utterances")
    def test_should_build_training_data_with_unknown_noise(
            self, mocked_augment_utterances, mocked_get_subtitles):
        # Given
        mocked_noises = ["mocked_noise_%s" % i for i in xrange(100)]
        mocked_get_subtitles.return_value = mocked_noises
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances

        num_intents = 3
        utterances_length = 5
        num_queries_per_intent = 3
        fake_utterance = {
            "data": [
                {"text": " ".join("1" for _ in xrange(utterances_length))}
            ]
        }
        dataset = {
            "intents": {
                unicode(i): {
                    "utterances": [fake_utterance] * num_queries_per_intent
                } for i in xrange(num_intents)
            }
        }
        random_state = np.random.RandomState(1)

        # When
        np.random.seed(42)
        noise_factor = 2
        replacement_string = "unknownword"
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            noise_factor=noise_factor, unknown_word_prob=0,
            unknown_words_replacement_string=replacement_string)
        utterances, _, intent_mapping = build_training_data(
            dataset, Language.EN, data_augmentation_config, random_state)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA])
                               for intent in dataset[INTENTS].values()
                               for utterance in intent[UTTERANCES]]
        np.random.seed(42)
        noise = list(mocked_noises)
        noise_size = int(min(noise_factor * num_queries_per_intent,
                             len(noise)))
        noisy_utterances = [replacement_string for _ in xrange(noise_size)]
        expected_utterances += list(noisy_utterances)
        expected_intent_mapping = dataset["intents"].keys() + [None]
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    def test_should_build_training_data_with_no_data(self):
        # Given
        language = Language.EN
        dataset = validate_and_format_dataset(get_empty_dataset(language))
        random_state = np.random.RandomState(1)

        # When
        data_augmentation_config = IntentClassifierConfig() \
            .data_augmentation_config
        utterances, _, intent_mapping = build_training_data(
            dataset, language, data_augmentation_config, random_state)

        # Then
        expected_utterances = []
        expected_intent_mapping = []
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noises")
    def test_generate_noise_utterances(self, mocked_get_noises):
        # Given
        language = Language.EN
        num_intents = 2
        noise_factor = 1
        utterances_length = 5

        noise = [unicode(i) for i in xrange(utterances_length)]
        mocked_get_noises.return_value = noise

        augmented_utterances = [
            {
                "data": [
                    {
                        "text": " ".join(
                            "{}".format(i) for i in xrange(utterances_length))
                    }
                ]
            }
        ]
        num_utterances = 10
        random_state = np.random.RandomState(1)

        augmented_utterances = augmented_utterances * num_utterances
        config = IntentClassifierDataAugmentationConfig(
            noise_factor=noise_factor)
        # When
        noise_utterances = generate_noise_utterances(
            augmented_utterances, num_intents, config, language, random_state)

        # Then
        joined_noise = " ".join(noise)
        for u in noise_utterances:
            self.assertEqual(u, joined_noise)

    def test_add_unknown_words_to_utterances(self):
        # Given
        utterances = [
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "dude",
                        "entity": "you"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": "dude",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "

                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    }
                ]
            }
        ]
        unknownword_prob = .5
        random_state = np.random.RandomState(1)

        # When
        replacement_string = "unknownword"
        noisy_utterances = add_unknown_word_to_utterances(
            utterances, unknown_word_prob=unknownword_prob,
            replacement_string=replacement_string, random_state=random_state
        )

        # Then
        expected_utterances = [
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " unknownword ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "dude",
                        "entity": "you"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": "unknownword",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": " unknownword ",
                        "entity": "you"
                    }
                ]
            }
        ]
        self.assertEqual(expected_utterances, noisy_utterances)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noises")
    def test_generate_noise_utterances_should_replace_unknown_words(
            self, mocked_noise):
        # Given
        utterances = [
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "bobby",
                        "entity": "you"
                    }
                ]
            }
        ]
        language = Language.EN
        mocked_noise.return_value = ["hello", "dear", "you", "fool"]
        replacement_string = "unknownword"

        # When
        noise = generate_smart_noise(utterances, replacement_string, language)

        # Then
        expected_noise = ["hello", replacement_string, "you",
                          replacement_string]
        self.assertEqual(noise, expected_noise)

    def test_remove_builtin_slots(self):
        # Given
        language = Language.EN
        dataset = {
            "snips_nlu_version": "0.0.1",
            "entities": {
                "snips/number": {}
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "I want ",
                                },
                                {
                                    "text": "three",
                                    "slot_name": "number_of_cups",
                                    "entity": "snips/number"
                                },
                                {
                                    "text": " cups",
                                },
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "give me ",
                                },
                                {
                                    "text": "twenty two",
                                    "slot_name": "number_of_cups",
                                    "entity": "snips/number"
                                },
                                {
                                    "text": " big cups please",
                                },
                            ]
                        }
                    ]
                }
            },
            "language": language.iso_code
        }

        # When
        filtered_dataset = remove_builtin_slots(dataset)

        # Then
        expected_dataset = {
            "snips_nlu_version": "0.0.1",
            "entities": {
                "snips/number": {}
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "I want ",
                                },
                                {
                                    "text": " cups",
                                },
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "give me ",
                                },
                                {
                                    "text": " big cups please",
                                },
                            ]
                        }
                    ]
                }
            },
            "language": language.iso_code
        }

        self.assertDictEqual(expected_dataset, filtered_dataset)
