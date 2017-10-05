# coding=utf-8
from __future__ import unicode_literals

import unittest

import numpy as np
from mock import patch

from snips_nlu.constants import INTENTS, LANGUAGE, DATA, UTTERANCES
from snips_nlu.dataset import validate_and_format_dataset, get_text_from_chunks
from snips_nlu.intent_classifier.feature_extraction import Featurizer
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier, build_training_data
from snips_nlu.languages import Language
from snips_nlu.tests.utils import SAMPLE_DATASET, empty_dataset


def np_random_permutation(x):
    return x


def get_mocked_augment_utterances(dataset, intent_name, language,
                                  max_utterances):
    return dataset[INTENTS][intent_name][UTTERANCES]


class TestSnipsIntentClassifier(unittest.TestCase):
    def test_intent_classifier_should_get_intent(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        classifier = SnipsIntentClassifier(language=Language.EN).fit(dataset)
        text = "This is a dummy_3 query from another intent"

        # When
        res = classifier.get_intent(text)
        intent = res[0]

        # Then
        expected_intent = "dummy_intent_2"

        self.assertEqual(intent, expected_intent)

    def test_should_get_none_if_empty_dataset(self):
        # Given
        dataset = empty_dataset(Language.EN)
        classifier = SnipsIntentClassifier(language=Language.EN).fit(dataset)
        text = "this is a dummy query"

        # When
        intent = classifier.get_intent(text)

        # Then
        expected_intent = None
        self.assertEqual(intent, expected_intent)

    @patch('snips_nlu.intent_classifier.feature_extraction.Featurizer.to_dict')
    def test_should_be_serializable(self, mock_to_dict):
        # Given
        mocked_dict = {"mocked_featurizer_key": "mocked_featurizer_value"}

        mock_to_dict.return_value = mocked_dict

        classifier_args = {
            "loss": 'log',
            "penalty": 'l2',
            "class_weight": 'balanced',
            "n_iter": 5,
            "random_state": 42,
            "n_jobs": -1
        }
        dataset = validate_and_format_dataset(SAMPLE_DATASET)

        intent_classifier = SnipsIntentClassifier(
            language=Language.EN, classifier_args=classifier_args).fit(dataset)
        coeffs = intent_classifier.classifier.coef_.tolist()
        intercept = intent_classifier.classifier.intercept_.tolist()

        # When
        classifier_dict = intent_classifier.to_dict()

        # Then
        intent_list = SAMPLE_DATASET[INTENTS].keys() + [None]
        expected_dict = {
            "classifier_args": classifier_args,
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": intent_list,
            "language_code": SAMPLE_DATASET[LANGUAGE],
            "featurizer": mocked_dict
        }
        self.assertEqual(classifier_dict, expected_dict)

    @patch('snips_nlu.intent_classifier.feature_extraction.Featurizer'
           '.from_dict')
    def should_be_deserializable(self, mock_from_dict):
        # Given
        mocked_featurizer = Featurizer(Language.EN)
        mock_from_dict.return_value = mocked_featurizer

        classifier_args = {
            "loss": 'log',
            "penalty": 'l2',
            "class_weight": 'balanced',
            "n_iter": 5,
            "random_state": 42,
            "n_jobs": -1
        }
        language = Language.EN
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

        classifier_dict = {
            "classifier_args": classifier_args,
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": intent_list,
            "language_code": language.iso_code,
            "featurizer": dict()
        }

        # When
        classifier = SnipsIntentClassifier.from_dict(classifier_dict)

        # Then
        self.assertEqual(classifier.language, language)
        self.assertDictEqual(classifier.classifier_args, classifier_args)
        self.assertEqual(classifier.intent_list, intent_list)
        self.assertIsNotNone(classifier.featurizer)
        self.assertListEqual(classifier.classifier.coef_.tolist(), coeffs)
        self.assertListEqual(classifier.classifier.intercept_.tolist(),
                             intercept)

    @patch("snips_nlu.intent_classifier.snips_intent_classifier"
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
                    "engineType": "regex",
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

        classifier_args = {
            "loss": 'log',
            "penalty": 'l2',
            "class_weight": 'balanced',
            "n_iter": 5,
            "random_state": 42,
            "n_jobs": -1
        }
        text = " "
        noise_size = 6
        utterance = [text] + [text] * noise_size
        labels = [1] + [None] * noise_size
        intent_list = ["dummy_intent_1", None]
        mocked_build_training.return_value = utterance, labels, intent_list

        # When / Then
        intent_classifier = SnipsIntentClassifier(
            language=Language.EN, classifier_args=classifier_args).fit(dataset)
        intent = intent_classifier.get_intent("no intent there")
        self.assertEqual(intent, None)

    @patch("snips_nlu.intent_classifier.snips_intent_classifier"
           ".augment_utterances")
    def test_should_build_training_data_with_no_stemming_no_noise(
            self, mocked_augment_utterances):
        # Given
        dataset = SAMPLE_DATASET
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances

        # When
        utterances, y, intent_mapping = build_training_data(
            dataset, Language.EN, use_stemming=False, noise_factor=0)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA]) for intent
                               in dataset[INTENTS].values() for utterance in
                               intent[UTTERANCES]]
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1']
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    @patch("snips_nlu.intent_classifier.snips_intent_classifier.get_subtitles")
    @patch("snips_nlu.intent_classifier.snips_intent_classifier"
           ".augment_utterances")
    def test_should_build_training_data_with_noise(
            self, mocked_augment_utterances, mocked_get_subtitles):
        # Given
        mocked_subtitles = set("mocked_subtitle_%s" % i for i in xrange(100))
        mocked_get_subtitles.return_value = mocked_subtitles
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances

        dataset = SAMPLE_DATASET
        nb_utterances = [len(intent[UTTERANCES]) for intent in
                         dataset[INTENTS].values()]
        avg_utterances = np.mean(nb_utterances)

        # When
        np.random.seed(42)
        noise_factor = 2
        utterances, y, intent_mapping = build_training_data(
            dataset, Language.EN, use_stemming=False,
            noise_factor=noise_factor)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA])
                               for intent in dataset[INTENTS].values()
                               for utterance in intent[UTTERANCES]]
        np.random.seed(42)
        noise = list(mocked_subtitles)
        noise_size = int(min(noise_factor * avg_utterances, len(noise)))
        noisy_utterances = np.random.choice(noise, size=noise_size,
                                            replace=False)
        expected_utterances += list(noisy_utterances)
        expected_intent_mapping = ['dummy_intent_2', 'dummy_intent_1', None]
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    def test_should_build_training_data_with_no_data(self):
        # Given
        language = Language.EN
        dataset = empty_dataset(language)

        # When
        utterances, y, intent_mapping = build_training_data(
            dataset, language, use_stemming=False, noise_factor=0)

        # Then
        expected_utterances = []
        expected_intent_mapping = []
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)
