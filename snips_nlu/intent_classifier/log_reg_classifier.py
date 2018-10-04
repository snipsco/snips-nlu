from __future__ import unicode_literals

import json
import logging
from builtins import range, str, zip
from pathlib import Path

import numpy as np
from future.utils import iteritems
from sklearn.linear_model import SGDClassifier

from snips_nlu.constants import LANGUAGE
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    build_training_data, get_regularization_factor, text_to_utterance)
from snips_nlu.pipeline.configs import LogRegIntentClassifierConfig
from snips_nlu.result import intent_classification_result
from snips_nlu.utils import (
    DifferedLoggingMessage, check_persisted_path, check_random_state,
    fitted_required, json_string, log_elapsed_time)

logger = logging.getLogger(__name__)

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
    def __init__(self, config=None, **shared):
        """The LogReg intent classifier can be configured by passing a
        :class:`.LogRegIntentClassifierConfig`"""
        if config is None:
            config = LogRegIntentClassifierConfig()
        super(LogRegIntentClassifier, self).__init__(config, **shared)
        self.classifier = None
        self.intent_list = None
        self.featurizer = None

    # pylint:enable=line-too-long

    @property
    def fitted(self):
        """Whether or not the intent classifier has already been fitted"""
        return self.intent_list is not None

    @log_elapsed_time(logger, logging.DEBUG,
                      "LogRegIntentClassifier in {elapsed_time}")
    def fit(self, dataset):
        """Fit the intent classifier with a valid Snips dataset

        Returns:
            :class:`LogRegIntentClassifier`: The same instance, trained
        """
        logger.debug("Fitting LogRegIntentClassifier...")
        dataset = validate_and_format_dataset(dataset)
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        language = dataset[LANGUAGE]
        random_state = check_random_state(self.config.random_seed)

        data_augmentation_config = self.config.data_augmentation_config
        utterances, classes, intent_list = build_training_data(
            dataset, language, data_augmentation_config, random_state)

        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self.featurizer = Featurizer(
            language,
            data_augmentation_config.unknown_words_replacement_string,
            self.config.featurizer_config,
            builtin_entity_parser=self.builtin_entity_parser,
            custom_entity_parser=self.custom_entity_parser
        )
        self.featurizer = self.featurizer.fit(dataset, utterances, classes)
        if self.featurizer is None:
            return self

        X = self.featurizer.transform(utterances)  # pylint: disable=C0103
        alpha = get_regularization_factor(dataset)
        self.classifier = SGDClassifier(random_state=random_state,
                                        alpha=alpha, **LOG_REG_ARGS)
        self.classifier.fit(X, classes)
        logger.debug("%s", DifferedLoggingMessage(self.log_best_features))
        return self

    @fitted_required
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
        if isinstance(intents_filter, str):
            intents_filter = [intents_filter]

        if not text or not self.intent_list \
                or self.featurizer is None or self.classifier is None:
            return None

        if len(self.intent_list) == 1:
            if self.intent_list[0] is None:
                return None
            return intent_classification_result(self.intent_list[0], 1.0)

        # pylint: disable=C0103
        X = self.featurizer.transform([text_to_utterance(text)])
        # pylint: enable=C0103
        proba_vec = self._predict_proba(X, intents_filter=intents_filter)
        intents_probas = sorted(zip(self.intent_list, proba_vec[0]),
                                key=lambda p: -p[1])
        for intent, proba in intents_probas:
            if intent is None:
                return None
            if intents_filter is None or intent in intents_filter:
                return intent_classification_result(intent, proba)
        return None

    def _predict_proba(self, X, intents_filter):  # pylint: disable=C0103
        self.classifier._check_proba()  # pylint: disable=W0212

        filtered_out_indexes = None
        if intents_filter is not None:
            filtered_out_indexes = [
                i for i, intent in enumerate(self.intent_list)
                if intent not in intents_filter and intent is not None]

        prob = self.classifier.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            if filtered_out_indexes:  # not None and not empty
                prob[:, filtered_out_indexes] = 0.
                # OvR normalization, like LibLinear's predict_probability
                prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            # We do not normalize when there is no intents filter, to keep the
            # probabilities calibrated
            return prob

    @check_persisted_path
    def persist(self, path):
        """Persist the object at the given path"""
        path = Path(path)
        path.mkdir()
        classifier_json = json_string(self.to_dict())
        with (path / "intent_classifier.json").open(mode="w") as f:
            f.write(classifier_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        """Load a :class:`LogRegIntentClassifier` instance from a path

        The data at the given path must have been generated using
        :func:`~LogRegIntentClassifier.persist`
        """
        path = Path(path)
        model_path = path / "intent_classifier.json"
        if not model_path.exists():
            raise OSError("Missing intent classifier model file: %s"
                          % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model_dict = json.load(f)
        return cls.from_dict(model_dict, **shared)

    @classmethod
    def from_dict(cls, unit_dict, **shared):
        """Creates a :class:`LogRegIntentClassifier` instance from a dict

        The dict must have been generated with
        :func:`~LogRegIntentClassifier.to_dict`
        """
        config = LogRegIntentClassifierConfig.from_dict(unit_dict["config"])
        intent_classifier = cls(config=config, **shared)
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
            intent_classifier.featurizer = Featurizer.from_dict(
                featurizer, **shared)
        return intent_classifier

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
            "config": self.config.to_dict(),
            "coeffs": coeffs,
            "intercept": intercept,
            "t_": t_,
            "intent_list": self.intent_list,
            "featurizer": featurizer_dict,
        }

    def log_best_features(self, top_n=20):
        log = "Top {} features weights by intent:\n".format(top_n)
        voca = {
            v: k for k, v in
            iteritems(self.featurizer.tfidf_vectorizer.vocabulary_)
        }
        features = [voca[i] for i in self.featurizer.best_features]
        for intent_ix in range(self.classifier.coef_.shape[0]):
            intent_name = self.intent_list[intent_ix]
            log += "\n\n\nFor intent {}\n".format(intent_name)
            top_features_idx = np.argsort(
                np.absolute(self.classifier.coef_[intent_ix]))[::-1][:top_n]
            for feature_ix in top_features_idx:
                feature_name = features[feature_ix]
                feature_weight = self.classifier.coef_[intent_ix, feature_ix]
                log += "\n{} -> {}".format(feature_name, feature_weight)
        return log
