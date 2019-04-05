from __future__ import unicode_literals

import json
import logging
from builtins import range, str, zip
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

from snips_nlu.common.log_utils import DifferedLoggingMessage, log_elapsed_time
from snips_nlu.common.utils import (
    check_persisted_path, check_random_state,
    fitted_required, json_string)
from snips_nlu.constants import LANGUAGE, RES_PROBA
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.exceptions import _EmptyDatasetUtterancesError, LoadingError
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    build_training_data, get_regularization_factor, text_to_utterance)
from snips_nlu.pipeline.configs import LogRegIntentClassifierConfig
from snips_nlu.result import intent_classification_result

logger = logging.getLogger(__name__)

LOG_REG_ARGS = {
    "loss": "log",
    "penalty": "l2",
    "class_weight": "balanced",
    "max_iter": 5,
    "n_jobs": -1
}


@IntentClassifier.register("log_reg_intent_classifier")
class LogRegIntentClassifier(IntentClassifier):
    """Intent classifier which uses a Logistic Regression underneath"""

    config_type = LogRegIntentClassifierConfig

    def __init__(self, config=None, **shared):
        """The LogReg intent classifier can be configured by passing a
        :class:`.LogRegIntentClassifierConfig`"""
        super(LogRegIntentClassifier, self).__init__(config, **shared)
        self.classifier = None
        self.intent_list = None
        self.featurizer = None

    @property
    def fitted(self):
        """Whether or not the intent classifier has already been fitted"""
        return self.intent_list is not None

    @log_elapsed_time(logger, logging.DEBUG,
                      "LogRegIntentClassifier in {elapsed_time}")
    def fit(self, dataset):
        """Fits the intent classifier with a valid Snips dataset

        Returns:
            :class:`LogRegIntentClassifier`: The same instance, trained
        """
        logger.debug("Fitting LogRegIntentClassifier...")
        dataset = validate_and_format_dataset(dataset)
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        language = dataset[LANGUAGE]
        random_state = check_random_state(self.config.random_seed)

        data_augmentation_config = self.config.data_augmentation_config
        utterances, classes, intent_list = build_training_data(
            dataset, language, data_augmentation_config, self.resources,
            random_state)

        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self.featurizer = Featurizer(
            config=self.config.featurizer_config,
            builtin_entity_parser=self.builtin_entity_parser,
            custom_entity_parser=self.custom_entity_parser,
            resources=self.resources
        )
        self.featurizer.language = language

        none_class = max(classes)
        try:
            x = self.featurizer.fit_transform(
                dataset, utterances, classes, none_class)
        except _EmptyDatasetUtterancesError:
            self.featurizer = None
            return self

        alpha = get_regularization_factor(dataset)
        self.classifier = SGDClassifier(random_state=random_state,
                                        alpha=alpha, **LOG_REG_ARGS)
        self.classifier.fit(x, classes)
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
            :class:`snips_nlu.exceptions.NotTrained`: When the intent
                classifier is not fitted

        """
        return self._get_intents(text, intents_filter)[0]

    @fitted_required
    def get_intents(self, text):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent

        Raises:
            :class:`snips_nlu.exceptions.NotTrained`: when the intent
                classifier is not fitted
        """
        return self._get_intents(text, intents_filter=None)

    def _get_intents(self, text, intents_filter):
        if isinstance(intents_filter, str):
            intents_filter = {intents_filter}
        elif isinstance(intents_filter, list):
            intents_filter = set(intents_filter)

        if not text or not self.intent_list or not self.featurizer:
            results = [intent_classification_result(None, 1.0)]
            results += [intent_classification_result(i, 0.0)
                        for i in self.intent_list if i is not None]
            return results

        if len(self.intent_list) == 1:
            return [intent_classification_result(self.intent_list[0], 1.0)]

        # pylint: disable=C0103
        X = self.featurizer.transform([text_to_utterance(text)])
        # pylint: enable=C0103
        proba_vec = self._predict_proba(X)
        logger.debug(
            "%s", DifferedLoggingMessage(self.log_activation_weights, text, X))
        results = [
            intent_classification_result(i, proba)
            for i, proba in zip(self.intent_list, proba_vec[0])
            if intents_filter is None or i is None or i in intents_filter]

        return sorted(results, key=lambda res: -res[RES_PROBA])

    def _predict_proba(self, X):  # pylint: disable=C0103
        self.classifier._check_proba()  # pylint: disable=W0212

        prob = self.classifier.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        return prob

    @check_persisted_path
    def persist(self, path):
        """Persists the object at the given path"""
        path.mkdir()

        featurizer = None
        if self.featurizer is not None:
            featurizer = "featurizer"
            featurizer_path = path / featurizer
            self.featurizer.persist(featurizer_path)

        coeffs = None
        intercept = None
        t_ = None
        if self.classifier is not None:
            coeffs = self.classifier.coef_.tolist()
            intercept = self.classifier.intercept_.tolist()
            t_ = self.classifier.t_

        self_as_dict = {
            "config": self.config.to_dict(),
            "coeffs": coeffs,
            "intercept": intercept,
            "t_": t_,
            "intent_list": self.intent_list,
            "featurizer": featurizer
        }

        classifier_json = json_string(self_as_dict)
        with (path / "intent_classifier.json").open(mode="w") as f:
            f.write(classifier_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`LogRegIntentClassifier` instance from a path

        The data at the given path must have been generated using
        :func:`~LogRegIntentClassifier.persist`
        """
        path = Path(path)
        model_path = path / "intent_classifier.json"
        if not model_path.exists():
            raise LoadingError("Missing intent classifier model file: %s"
                               % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model_dict = json.load(f)

        # Create the classifier
        config = LogRegIntentClassifierConfig.from_dict(model_dict["config"])
        intent_classifier = cls(config=config, **shared)
        intent_classifier.intent_list = model_dict['intent_list']

        # Create the underlying SGD classifier
        sgd_classifier = None
        coeffs = model_dict['coeffs']
        intercept = model_dict['intercept']
        t_ = model_dict["t_"]
        if coeffs is not None and intercept is not None:
            sgd_classifier = SGDClassifier(**LOG_REG_ARGS)
            sgd_classifier.coef_ = np.array(coeffs)
            sgd_classifier.intercept_ = np.array(intercept)
            sgd_classifier.t_ = t_
        intent_classifier.classifier = sgd_classifier

        # Add the featurizer
        featurizer = model_dict['featurizer']
        if featurizer is not None:
            featurizer_path = path / featurizer
            intent_classifier.featurizer = Featurizer.from_path(
                featurizer_path, **shared)

        return intent_classifier

    def log_best_features(self, top_n=50):
        if not hasattr(self.featurizer, "feature_index_to_feature_name"):
            return None

        log = "Top {} features weights by intent:".format(top_n)
        index_to_feature = self.featurizer.feature_index_to_feature_name
        for intent_ix in range(self.classifier.coef_.shape[0]):
            intent_name = self.intent_list[intent_ix]
            log += "\n\n\nFor intent {}\n".format(intent_name)
            top_features_idx = np.argsort(
                np.absolute(self.classifier.coef_[intent_ix]))[::-1][:top_n]
            for feature_ix in top_features_idx:
                feature_name = index_to_feature[feature_ix]
                feature_weight = self.classifier.coef_[intent_ix, feature_ix]
                log += "\n{} -> {}".format(feature_name, feature_weight)
        return log

    def log_activation_weights(self, text, x, top_n=50):
        if not hasattr(self.featurizer, "feature_index_to_feature_name"):
            return None

        log = "\n\nTop {} feature activations for: \"{}\":\n".format(
            top_n, text)
        activations = np.multiply(
            self.classifier.coef_, np.asarray(x.todense()))
        abs_activation = np.absolute(activations).flatten().squeeze()

        if top_n > activations.size:
            top_n = activations.size

        top_n_activations_ix = np.argpartition(abs_activation, -top_n,
                                               axis=None)[-top_n:]
        top_n_activations_ix = np.unravel_index(
            top_n_activations_ix, activations.shape)

        index_to_feature = self.featurizer.feature_index_to_feature_name
        features_intent_and_activation = [
            (self.intent_list[i], index_to_feature[f], activations[i, f])
            for i, f in zip(*top_n_activations_ix)]

        features_intent_and_activation = sorted(
            features_intent_and_activation, key=lambda x: abs(x[2]),
            reverse=True)

        for intent, feature, activation in features_intent_and_activation:
            log += "\n\n\"{}\" -> ({}, {:.2f})".format(
                intent, feature, float(activation))
        log += "\n\n"
        return log
