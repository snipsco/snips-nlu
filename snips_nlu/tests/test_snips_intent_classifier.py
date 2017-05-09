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
from snips_nlu.utils import CLASS_NAME, MODULE_NAME, safe_pickle_dumps


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

    def test_should_get_none_if_only_builtin_intents(self):
        # Given
        language = Language.EN
        dataset = {
            "intents": {
                "GetWeather": {
                    "engineType": "tensorflow",
                    "utterances": [
                        {
                            "data": "Is the weather gonna get better next"
                                    " week in Paris?"
                        }
                    ]
                }
            },
            "entities": {},
            "language": language.iso_code
        }
        classifier = SnipsIntentClassifier(language=Language.EN).fit(dataset)
        text = "this is a dummy query"

        # When
        intent = classifier.get_intent(text)

        # Then
        expected_intent = None
        self.assertEqual(intent, expected_intent)

    @patch(
        'snips_nlu.intent_classifier.feature_extraction.Featurizer.from_dict')
    @patch(
        'snips_nlu.intent_classifier.feature_extraction.Featurizer.to_dict')
    def test_should_be_serializable(self, mocked_featurizer_to_dict,
                                    mocked_featurizer_from_dict):
        # Given
        def mock_to_dict():
            return {"mocked_featurizer_key": "mocked_featurizer_value"}

        mocked_featurizer_to_dict.side_effect = mock_to_dict

        def mock_from_dict(_):
            return Featurizer(Language.EN)

        mocked_featurizer_from_dict.side_effect = mock_from_dict

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
        coeffs = intent_classifier.classifier.coef_
        intercept = intent_classifier.classifier.intercept_

        # When
        classifier_dict = intent_classifier.to_dict()

        # Then
        # noinspection PyBroadException
        try:
            dumped = json.dumps(classifier_dict).encode("utf-8")
        except:
            self.fail("SnipsIntentClassifier dict should be json serializable "
                      "to utf-8")

        # noinspection PyBroadException
        try:
            _ = SnipsIntentClassifier.from_dict(json.loads(dumped))
        except:
            self.fail("SnipsIntentClassifier should be deserializable from "
                      "dict with unicode values")

        intent_list = SAMPLE_DATASET[INTENTS].keys() + [None]
        expected_dict = {
            CLASS_NAME: SnipsIntentClassifier.__name__,
            MODULE_NAME: SnipsIntentClassifier.__module__,
            "classifier_args": classifier_args,
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": intent_list,
            "language_code": SAMPLE_DATASET[LANGUAGE],
            "featurizer": mock_to_dict()
        }
        self.assertEqual(classifier_dict["intent_list"],
                         expected_dict["intent_list"])

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
