# coding=utf-8
from __future__ import unicode_literals

import unittest

from snips_nlu.config import CRFFeaturesConfig
from snips_nlu.constants import ENTITIES
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.languages import Language
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.slot_filler.default.default_features_functions import \
    compute_entity_collection_size
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
        features = engine.probabilistic_parser.crf_taggers[intent].features
        builtin_features_count = len(
            [f for f in features if "built-in-snips/number" in f])
        self.assertGreater(builtin_features_count, 0)

    def test_compute_entity_collection_size(self):
        # Given
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        crf_features_config = CRFFeaturesConfig(base_drop_ratio=.4)
        collection = dataset[ENTITIES]["Temperature"].keys()

        expected_collection_size = 1

        # When / Then

        self.assertEqual(
            expected_collection_size,
            compute_entity_collection_size(collection, crf_features_config))
