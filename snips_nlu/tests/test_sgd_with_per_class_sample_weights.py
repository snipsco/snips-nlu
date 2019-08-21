import unittest

from future.builtins import str
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier

from snips_nlu.intent_classifier.sgd_with_per_class_sample_weights import (
    SGDClassifierWithSampleWeightsPerClass)




class TestSGDClassifierWithSampleWeightsPerClass(unittest.TestCase):
    def test_validate_sample_weight_should_raise(self):
        # Given
        clf = SGDClassifierWithSampleWeightsPerClass()

        x, y = make_blobs(n_samples=10, n_features=5, centers=3)
        sample_weight = [
            [1, 1, 0],
            [1, 0, 1],
        ]

        # When / Then
        expected_msg = "Expected sample_weight to be of shape" \
                       " (3, 10), found (2, 3)"
        with self.assertRaises(ValueError) as ctx:
            clf.fit(x, y, sample_weight=sample_weight)

        msg = str(ctx.exception)
        self.assertEqual(expected_msg, msg)

    def test_should_behave_like_sgd_when_no_weights(self):
        # Given
        random_state = 42
        x, y = make_blobs(n_samples=10, n_features=5, centers=3)
        clf_1 = SGDClassifierWithSampleWeightsPerClass(
            random_state=random_state)
        clf_2 = SGDClassifier(random_state=random_state)

        # When
        clf_1.fit(x, y)
        clf_2.fit(x, y)

        # Then
        self.assertEqual(clf_1.intercept_.tolist(), clf_2.intercept_.tolist())
        self.assertEqual(clf_1.coef_.tolist(), clf_2.coef_.tolist())
