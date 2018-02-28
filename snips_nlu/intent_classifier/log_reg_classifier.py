from __future__ import unicode_literals

from builtins import str, zip

import numpy as np
from sklearn.linear_model import SGDClassifier

from snips_nlu.constants import LANGUAGE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.log_reg_classifier_utils import \
    remove_builtin_slots, get_regularization_factor, build_training_data
from snips_nlu.pipeline.configs import LogRegIntentClassifierConfig
from snips_nlu.result import intent_classification_result
from snips_nlu.utils import check_random_state, NotTrained

LOG_REG_ARGS = {
    "loss": "log",
    "penalty": "l2",
    "class_weight": "balanced",
    "max_iter": 5,
    "n_jobs": -1
}


class LogRegIntentClassifier(IntentClassifier):
    """Intent classifier which uses a Logistic Regression underneath"""

    unit_name = "log_reg_intent_classifier"
    config_type = LogRegIntentClassifierConfig

    # pylint:disable=line-too-long
    def __init__(self, config=None):
        """The LogReg intent classifier can be configured by passing a
        :class:`.LogRegIntentClassifierConfig`"""
        if config is None:
            config = LogRegIntentClassifierConfig()
        super(LogRegIntentClassifier, self).__init__(config)
        self.classifier = None
        self.intent_list = None
        self.featurizer = None

    # pylint:enable=line-too-long

    @property
    def fitted(self):
        """Whether or not the intent classifier has already been fitted"""
        return self.intent_list is not None

    def fit(self, dataset):
        """Fit the intent classifier with a valid Snips dataset

        Returns:
            :class:`LogRegIntentClassifier`: The same instance, trained
        """
        dataset = validate_and_format_dataset(dataset)
        language = dataset[LANGUAGE]
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

    def get_intent(self, text, intents_filter=None):
        """Performs intent classification on the provided *text*

        Args:
            text (str): Input
            intents_filter (str or list of str): When defined, it will find
                the most likely intent among the list, otherwise it will use
                the whole list of intents defined in the dataset

        Returns:
            dict or None: The most likely intent along with its probability or
            *None* if no intent was found

        Raises:
            NotTrained: When the intent classifier is not fitted

        """
        if not self.fitted:
            raise NotTrained('LogRegIntentClassifier must be fitted')

        if isinstance(intents_filter, str):
            intents_filter = [intents_filter]

        if not text or not self.intent_list \
                or self.featurizer is None or self.classifier is None:
            return None

        if len(self.intent_list) == 1:
            if self.intent_list[0] is None:
                return None
            return intent_classification_result(self.intent_list[0], 1.0)

        X = self.featurizer.transform([text])  # pylint: disable=C0103
        proba_vec = self.classifier.predict_proba(X)[0]
        intents_probas = sorted(zip(self.intent_list, proba_vec),
                                key=lambda p: -p[1])
        for intent, proba in intents_probas:
            if intent is None:
                return None
            if intents_filter is None or intent in intents_filter:
                return intent_classification_result(intent, proba)
        return None

    def to_dict(self):
        """Returns a json-serializable dict"""
        featurizer_dict = None
        if self.featurizer is not None:
            featurizer_dict = self.featurizer.to_dict()
        coeffs = None
        intercept = None
        t_ = None
        if self.classifier is not None:
            coeffs = self.classifier.coef_.tolist()
            intercept = self.classifier.intercept_.tolist()
            t_ = self.classifier.t_

        return {
            "unit_name": self.unit_name,
            "config": self.config.to_dict(),
            "coeffs": coeffs,
            "intercept": intercept,
            "t_": t_,
            "intent_list": self.intent_list,
            "featurizer": featurizer_dict,
        }

    @classmethod
    def from_dict(cls, unit_dict):
        """Creates a :class:`LogRegIntentClassifier` instance from a dict

        The dict must have been generated with
        :func:`~LogRegIntentClassifier.to_dict`
        """
        config = LogRegIntentClassifierConfig.from_dict(unit_dict["config"])
        intent_classifier = cls(config=config)
        sgd_classifier = None
        coeffs = unit_dict['coeffs']
        intercept = unit_dict['intercept']
        t_ = unit_dict["t_"]
        if coeffs is not None and intercept is not None:
            sgd_classifier = SGDClassifier(**LOG_REG_ARGS)
            sgd_classifier.coef_ = np.array(coeffs)
            sgd_classifier.intercept_ = np.array(intercept)
            sgd_classifier.t_ = t_
        intent_classifier.classifier = sgd_classifier
        intent_classifier.intent_list = unit_dict['intent_list']
        featurizer = unit_dict['featurizer']
        if featurizer is not None:
            intent_classifier.featurizer = Featurizer.from_dict(featurizer)
        return intent_classifier
