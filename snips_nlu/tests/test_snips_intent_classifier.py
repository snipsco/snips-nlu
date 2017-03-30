import unittest

from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.tests.utils import EMPTY_DATASET, SAMPLE_DATASET
from snips_nlu.languages import Language

class TestSnipsIntentClassifier(unittest.TestCase):
    def test_should_get_intent(self):
        # Given
        dataset = SAMPLE_DATASET
        language = 'eng'
        language = Language.from_iso_code(language)
        classifier = SnipsIntentClassifier(language=language).fit(dataset)
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
        language = 'eng'
        language = Language.from_iso_code(language)
        classifier = SnipsIntentClassifier(language=language).fit(dataset)
        text = "this is a dummy query"

        # When
        intent = classifier.get_intent(text)

        # Then
        expected_intent = None
        self.assertEqual(intent, expected_intent)


if __name__ == '__main__':
    unittest.main()
