import numpy as np
from sklearn.linear_model import SGDClassifier

from data_augmentation import build_training_data, get_regularization_factor
from feature_extraction import Featurizer
from intent_classifier import IntentClassifier
from snips_nlu.constants import CUSTOM_ENGINE, BUILTIN_ENGINE
from snips_nlu.dataset import filter_dataset
from snips_nlu.languages import Language
from snips_nlu.preprocessing import stem_sentence
from snips_nlu.result import IntentClassificationResult
from snips_nlu.utils import (instance_to_generic_dict, ensure_string,
                             safe_pickle_dumps, safe_pickle_loads)


def get_default_parameters():
    return {
        "loss": 'log',
        "penalty": 'l2',
        "class_weight": 'balanced',
        "n_iter": 5,
        "random_state": 42,
        "n_jobs": -1
    }


class SnipsIntentClassifier(IntentClassifier):
    def __init__(self, language, classifier_args=get_default_parameters()):
        self.language = language
        self.classifier_args = classifier_args
        self.classifier = None
        self.intent_list = None
        self.featurizer = Featurizer(self.language)

    @property
    def fitted(self):
        return self.intent_list is not None

    def fit(self, dataset):
        min_utterances = 1
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE, min_utterances)
        builtin_dataset = filter_dataset(dataset, BUILTIN_ENGINE,
                                         min_utterances)
        utterances, y, intent_list = build_training_data(custom_dataset,
                                                         builtin_dataset,
                                                         self.language)
        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self.featurizer = self.featurizer.fit(utterances, y)
        if self.featurizer is None:
            return self

        X = self.featurizer.transform(utterances)
        alpha = get_regularization_factor(custom_dataset)
        self.classifier_args.update({'alpha': alpha})
        self.classifier = SGDClassifier(**self.classifier_args).fit(X, y)
        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError('SnipsIntentClassifier instance must be '
                                 'fitted before `get_intent` can be called')

        if len(text) == 0 or len(self.intent_list) == 0:
            return None

        if len(text) == 0 or len(self.intent_list) == 0 or \
                        self.featurizer is None:
            return None

        if len(self.intent_list) == 1:
            if self.intent_list[0] is None:
                return None
            return IntentClassificationResult(self.intent_list[0], 1.0)

        stemmed_text = stem_sentence(text, self.language)

        X = self.featurizer.transform([stemmed_text])
        proba_vect = self.classifier.predict_proba(X)
        predicted = np.argmax(proba_vect[0])

        intent_name = self.intent_list[int(predicted)]
        prob = proba_vect[0][int(predicted)]

        if intent_name is None:
            return None

        return IntentClassificationResult(intent_name, prob)

    def to_dict(self):
        return {
            "classifier_args": self.classifier_args,
            "coeffs": self.classifier.coef_.tolist(),
            "intercept": self.classifier.intercept_.tolist(),
            "intent_list": self.intent_list,
            "language_code": self.language.iso_code,
            "featurizer": self.featurizer.to_dict()
        }

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict['language_code'])
        classifier_args = obj_dict['classifier_args']
        classifier = cls(language=language, classifier_args=classifier_args)
        sgd_classifier = SGDClassifier(**classifier_args)
        sgd_classifier.coef_ = np.array(obj_dict['coeffs'])
        sgd_classifier.intercept_ = np.array(obj_dict['intercept'])
        classifier.classifier = sgd_classifier
        classifier.intent_list = obj_dict['intent_list']
        classifier.featurizer = Featurizer.from_dict(obj_dict['featurizer'])
        return classifier
