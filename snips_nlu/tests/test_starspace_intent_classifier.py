# coding=utf-8
from __future__ import unicode_literals

import json
import logging
import shutil

from snips_nlu.constants import ROOT_PATH
from snips_nlu.intent_classifier.starspace_intent_classifier import (
    StarSpaceIntentClassifier, StarSpaceIntentClassifierConfig)
from snips_nlu.pipeline.configs import IntentClassifierDataAugmentationConfig
from snips_nlu.tests.utils import SnipsTest

snips_nlu_logger = logging.getLogger("snips_nlu")

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

snips_nlu_logger.addHandler(handler)
snips_nlu_logger.setLevel(logging.DEBUG)


class TestStarSpaceIntentClassifier(SnipsTest):
    def test_startspace_intent_classifier(self):
        # Given
        path = ROOT_PATH / "assistant" / "dataset.json"
        with path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

        # dataset = BEVERAGE_DATASET
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            min_utterances=200, noise_factor=5)
        config = StarSpaceIntentClassifierConfig(
            validation_patience=20, margin=0.3,
            data_augmentation_config=data_augmentation_config,
            embedding_dim=10, max_intents=2, use_stemming=True)
        classifier = StarSpaceIntentClassifier(config)
        text = "make me some hot tea with 3 coffees please"

        # When
        res = classifier.fit(dataset).get_intent(text)
        classifier_path = ROOT_PATH / "starspace_classifier"
        if classifier_path.exists():
            shutil.rmtree(str(classifier_path))
        classifier.persist(classifier_path)

        builtin_entity_parser_path = ROOT_PATH / "builtin_entity_parser"
        custom_entity_parser_path = ROOT_PATH / "custom_entity_parser"

        if builtin_entity_parser_path.exists():
            shutil.rmtree(str(builtin_entity_parser_path))
        if custom_entity_parser_path.exists():
            shutil.rmtree(str(custom_entity_parser_path))

        classifier.builtin_entity_parser.persist(builtin_entity_parser_path)
        classifier.custom_entity_parser.persist(custom_entity_parser_path)

        # Then
        print("Result -> ", res)
