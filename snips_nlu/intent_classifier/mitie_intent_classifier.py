# coding=utf-8
from __future__ import unicode_literals

from collections import namedtuple

import mitie
from builtins import str, zip
from sklearn.utils import check_random_state

from snips_nlu.constants import LANGUAGE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.log_reg_classifier_utils import \
    build_training_data
from snips_nlu.pipeline.configs import IntentClassifierDataAugmentationConfig, \
    MitieIntentClassifierConfig
from snips_nlu.result import intent_classification_result


class MitieIntentClassifier(IntentClassifier):
    unit_name = "mitie_intent_classifier"
    config_type = MitieIntentClassifierConfig

    def __init__(self, config, **shared):
        super(MitieIntentClassifier, self).__init__(config, **shared)
        self.feature_extractor_file = "total_word_feature_extractor.dat"
        self.feature_extractor = mitie.total_word_feature_extractor(
            str(self.feature_extractor_file))
        self.config = namedtuple(
            "data_augmentation_config",
            IntentClassifierDataAugmentationConfig())

    def fit(self, dataset):
        dataset = validate_and_format_dataset(dataset)
        language = dataset[LANGUAGE]

        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        random_state = check_random_state(self.config.random_seed)
        data_augmentation_config = self.config.data_augmentation_config
        utterances, classes, intent_list = build_training_data(
            dataset, language, data_augmentation_config, random_state)

        trainer = mitie.text_categorizer_trainer(
            str(self.feature_extractor_file))

        trainer.num_threads = 4
        for u, c in zip(utterances, classes):
            tokens = mitie.tokenize(u)
            trainer.add_labeled_text(tokens, str(c))
        self.classifier = trainer.train()

    def get_intent(self, text, intents_filter=None):
        tokens = mitie.tokenize(text)
        intent, prob = self.classifier(tokens, self.feature_extractor)
        if intent == "None":
            intent = None
        return intent_classification_result(intent, prob)
