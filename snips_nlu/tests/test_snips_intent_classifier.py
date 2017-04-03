import json
import unittest

from mock import patch

from snips_nlu.constants import INTENTS, LANGUAGE
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.tests.utils import EMPTY_DATASET, SAMPLE_DATASET
from snips_nlu.utils import CLASS_NAME, MODULE_NAME


class TestSnipsIntentClassifier(unittest.TestCase):
    def test_should_get_intent(self):
        # Given
        dataset = SAMPLE_DATASET
        classifier = SnipsIntentClassifier().fit(dataset)
        text = "This is a dummy_3 query from another intent"

        # When
        res = classifier.get_intent(text)
        intent = res[0]

        # Then
        expected_intent = "dummy_intent_2"

        self.assertEqual(intent, expected_intent)

    def test_should_get_none_if_empty_dataset(self):
        # Given
        dataset = EMPTY_DATASET
        classifier = SnipsIntentClassifier().fit(dataset)
        text = "this is a dummy query"

        # When
        intent = classifier.get_intent(text)

        # Then
        expected_intent = None
        self.assertEqual(intent, expected_intent)

    @patch('cPickle.dumps')
    @patch('snips_nlu.intent_classifier.feature_extraction.Featurizer.to_dict')
    def test_should_be_serializable(self, mocked_featurizer_to_dict,
                                    mocked_pickle):
        # Given
        def mock_to_dict():
            return {"mocked_featurizer_key": "mocked_featurizer_value"}

        def mock_pickle_dumps(obj):
            return "mocked_pkl_string"

        mocked_featurizer_to_dict.side_effect = mock_to_dict
        mocked_pickle.side_effect = mock_pickle_dumps

        classifier_args = {
            "loss": 'log',
            "penalty": 'l2',
            "class_weight": 'balanced',
            "n_iter": 5,
            "random_state": 42,
            "n_jobs": -1
        }

        intent_classifier = SnipsIntentClassifier(
            classifier_args=classifier_args).fit(SAMPLE_DATASET)

        # When
        classifier_dict = intent_classifier.to_dict()

        # Then
        try:
            json.dumps(classifier_dict)
        except:
            self.fail("IntentClassifier dict should be json serializable")

        intent_list = [None] + SAMPLE_DATASET[INTENTS].keys()
        expected_dict = {
            CLASS_NAME: SnipsIntentClassifier.__name__,
            MODULE_NAME: SnipsIntentClassifier.__module__,
            "classifier_args": classifier_args,
            "classifier_pkl": "mocked_pkl_string",
            "intent_list": intent_list,
            "language_code": SAMPLE_DATASET[LANGUAGE],
            "featurizer": mock_to_dict()
        }
        self.assertDictEqual(classifier_dict, expected_dict)


if __name__ == '__main__':
    unittest.main()
