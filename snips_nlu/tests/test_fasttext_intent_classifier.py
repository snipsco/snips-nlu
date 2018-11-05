# coding=utf-8
from __future__ import unicode_literals

import json
import logging
import shutil

from snips_nlu.constants import ROOT_PATH
from snips_nlu.intent_classifier.fastext_intent_classifier import (
    FastTextIntentClassifier)
from snips_nlu.pipeline.configs import IntentClassifierDataAugmentationConfig
from snips_nlu.pipeline.configs.intent_classifier import \
    FastTextIntentClassifierConfig
from snips_nlu.tests.utils import SnipsTest

snips_nlu_logger = logging.getLogger("snips_nlu")

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

snips_nlu_logger.addHandler(handler)
snips_nlu_logger.setLevel(logging.DEBUG)


class TestFastTextIntentClassifier(SnipsTest):
    def test_fasttext_intent_classifier(self):
        # Given
        path = ROOT_PATH / "dataset.json"
        with path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

        # dataset = BEVERAGE_DATASET
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            min_utterances=200, noise_factor=5)
        config = FastTextIntentClassifierConfig(
            data_augmentation_config=data_augmentation_config,
            use_stemming=True)
        classifier = FastTextIntentClassifier(config)
        text = ""

        # When
        res = classifier.fit(dataset).get_intent(text)
        classifier_path = ROOT_PATH / "fasttext_classifier"
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
