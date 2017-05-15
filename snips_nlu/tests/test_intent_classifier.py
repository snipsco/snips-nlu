from __future__ import unicode_literals

import json
import unittest

from mock import patch

from snips_nlu.constants import INTENTS, LANGUAGE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier.feature_extraction import Featurizer
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.languages import Language
from snips_nlu.tests.utils import SAMPLE_DATASET, empty_dataset


class TestSnipsIntentClassifier(unittest.TestCase):
    def test_should_get_intent(self):
        # Given
        dataset = SAMPLE_DATASET
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

        intent_classifier = SnipsIntentClassifier(
            language=Language.EN, classifier_args=classifier_args).fit(
            SAMPLE_DATASET)
        coeffs = intent_classifier.classifier.coef_.tolist()
        intercept = intent_classifier.classifier.intercept_.tolist()

        # When
        classifier_dict = intent_classifier.to_dict()

        # Then
        # noinspection PyBroadException
        try:
            json.dumps(classifier_dict).encode("utf-8")
        except:
            self.fail("SnipsIntentClassifier dict should be json serializable "
                      "to utf-8")

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


if __name__ == '__main__':
    unittest.main()
