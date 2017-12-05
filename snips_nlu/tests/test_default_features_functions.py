# coding=utf-8
from __future__ import unicode_literals

import unittest

from snips_nlu.languages import Language
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.tests.utils import BEVERAGE_DATASET


class TestDefaultFeaturesFunction(unittest.TestCase):
    def test_should_include_builtin_features(self):
        # Given
        dataset = BEVERAGE_DATASET
        language = Language.EN
        engine = SnipsNLUEngine(language)
        intent = "MakeCoffee"

        # When
        engine = engine.fit(dataset)

        # Then
        features = engine.probabilistic_parser.slot_fillers[intent].features
        builtin_features_count = len(
            [f for f in features if "built-in-snips/number" in f.name])
        self.assertGreater(builtin_features_count, 0)
