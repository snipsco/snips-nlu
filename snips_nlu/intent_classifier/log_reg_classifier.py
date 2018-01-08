from __future__ import unicode_literals

import numpy as np
from sklearn.linear_model import SGDClassifier

from snips_nlu.constants import (LANGUAGE)
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.log_reg_classifier_utils import \
    remove_builtin_slots, get_regularization_factor, build_training_data
from snips_nlu.languages import Language
from snips_nlu.pipeline.configs.intent_classifier import \
    IntentClassifierConfig
from snips_nlu.result import IntentClassificationResult
from snips_nlu.utils import check_random_state

LOG_REG_ARGS = {
    "loss": "log",
    "penalty": "l2",
    "class_weight": "balanced",
    "n_iter": 5,
    "n_jobs": -1
}


class LogRegIntentClassifier(IntentClassifier):
    unit_name = "log_reg_intent_classifier"
    config_type = IntentClassifierConfig

    def __init__(self, config=None):
        if config is None:
            config = IntentClassifierConfig()
        super(LogRegIntentClassifier, self).__init__(config)
        self.classifier = None
        self.intent_list = None
        self.featurizer = None

    @property
    def fitted(self):
        return self.intent_list is not None

    def fit(self, dataset):
        language = Language.from_iso_code(dataset[LANGUAGE])
        random_state = check_random_state(self.config.random_seed)
        filtered_dataset = remove_builtin_slots(dataset)
        data_augmentation_config = self.config.data_augmentation_config
        utterances, y, intent_list = build_training_data(
            filtered_dataset, language, data_augmentation_config, random_state)

        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self.featurizer = Featurizer(
            language,
            data_augmentation_config.unknown_words_replacement_string,
            self.config.featurizer_config)
        self.featurizer = self.featurizer.fit(filtered_dataset, utterances, y)
        if self.featurizer is None:
            return self

        X = self.featurizer.transform(utterances)  # pylint: disable=C0103
        alpha = get_regularization_factor(filtered_dataset)
        self.classifier = SGDClassifier(random_state=random_state,
                                        alpha=alpha, **LOG_REG_ARGS)
        self.classifier.fit(X, y)
        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError('LogRegIntentClassifier instance must be '
                                 'fitted before `get_intent` can be called')

        if not text or not self.intent_list \
                or self.featurizer is None or self.classifier is None:
            return None

        if len(self.intent_list) == 1:
            if self.intent_list[0] is None:
                return None
            return IntentClassificationResult(self.intent_list[0], 1.0)

        X = self.featurizer.transform([text])  # pylint: disable=C0103
        proba_vect = self.classifier.predict_proba(X)
        predicted = np.argmax(proba_vect[0])

        intent_name = self.intent_list[int(predicted)]
        prob = proba_vect[0][int(predicted)]

        if intent_name is None:
            return None

        return IntentClassificationResult(intent_name, prob)

    def to_dict(self):
        featurizer_dict = None
        if self.featurizer is not None:
            featurizer_dict = self.featurizer.to_dict()
        coeffs = None
        intercept = None
        if self.classifier is not None:
            coeffs = self.classifier.coef_.tolist()
            intercept = self.classifier.intercept_.tolist()

        return {
            "unit_name": self.unit_name,
            "config": self.config.to_dict(),
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": self.intent_list,
            "featurizer": featurizer_dict,
        }

    @classmethod
    def from_dict(cls, unit_dict):
        config = IntentClassifierConfig.from_dict(unit_dict["config"])
        intent_classifier = cls(config=config)
        sgd_classifier = None
        coeffs = unit_dict['coeffs']
        intercept = unit_dict['intercept']
        if coeffs is not None and intercept is not None:
            sgd_classifier = SGDClassifier(**LOG_REG_ARGS)
            sgd_classifier.coef_ = np.array(coeffs)
            sgd_classifier.intercept_ = np.array(intercept)
        intent_classifier.classifier = sgd_classifier
        intent_classifier.intent_list = unit_dict['intent_list']
        featurizer = unit_dict['featurizer']
        if featurizer is not None:
            intent_classifier.featurizer = Featurizer.from_dict(featurizer)
        return intent_classifier
