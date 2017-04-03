import unittest

from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier, get_default_parameters
from snips_nlu.languages import Language
from snips_nlu.tests.utils import EMPTY_DATASET, SAMPLE_DATASET


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

    def test_should_be_serializable(self):
        # Given
        language = Language.ENG
        classifier_args = {
            "loss": 'log',
            "penalty": 'l2',
            "class_weight": 'balanced',
            "n_iter": 5,
            "random_state": 42,
            "n_jobs": -1
        }

        classifier = SnipsIntentClassifier()


if __name__ == '__main__':
    unittest.main()
