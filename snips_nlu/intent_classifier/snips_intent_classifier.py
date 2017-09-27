from __future__ import unicode_literals

from itertools import izip
from uuid import uuid4

import numpy as np
from sklearn.linear_model import SGDClassifier

from feature_extraction import Featurizer
from snips_nlu.constants import INTENTS, UTTERANCES, DATA
from snips_nlu.data_augmentation import augment_utterances, \
    DataAugmentationConfig
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.languages import Language
from snips_nlu.resources import get_subtitles
from snips_nlu.result import IntentClassificationResult

NOISE_NAME = str(uuid4()).decode()


def get_regularization_factor(dataset):
    intents = dataset[INTENTS]
    nb_utterances = [len(intent[UTTERANCES]) for intent in intents.values()]
    avg_utterances = np.mean(nb_utterances)
    total_utterances = sum(nb_utterances)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def build_training_data(dataset, language, noise_factor=5, use_stemming=True,
                        min_utterances_per_intent=None):
    # Creating class mapping
    intents = dataset[INTENTS]
    intent_index = 0
    classes_mapping = dict()
    for intent in intents:
        classes_mapping[intent] = intent_index
        intent_index += 1

    noise_class = intent_index

    # Computing dataset statistics
    nb_utterances = [len(intent[UTTERANCES]) for intent in intents.values()]

    augmented_utterances = []
    utterance_classes = []
    for nb_utterance, intent_name in izip(nb_utterances, intents.keys()):
        min_utterances_to_generate = max(min_utterances_per_intent,
                                         nb_utterance)
        config = DataAugmentationConfig(
            max_utterances=min_utterances_to_generate)
        utterances = augment_utterances(
            dataset, intent_name, language=language, **config.to_dict())
        augmented_utterances += [get_text_from_chunks(utterance[DATA]) for
                                 utterance in utterances]
        utterance_classes += [classes_mapping[intent_name] for _ in utterances]

    # Adding noise
    avg_utterances = np.mean(nb_utterances) if len(nb_utterances) > 0 else 0
    noise = list(get_subtitles(language))
    noise_size = min(int(noise_factor * avg_utterances), len(noise))
    noisy_utterances = np.random.choice(noise, size=noise_size, replace=False)
    augmented_utterances += list(noisy_utterances)
    utterance_classes += [noise_class for _ in noisy_utterances]
    if len(noisy_utterances) > 0:
        classes_mapping[NOISE_NAME] = noise_class

    nb_classes = len(set(classes_mapping.values()))
    intent_mapping = [None for _ in xrange(nb_classes)]
    for intent, intent_class in classes_mapping.iteritems():
        if intent == NOISE_NAME:
            intent_mapping[intent_class] = None
        else:
            intent_mapping[intent_class] = intent

    return augmented_utterances, np.array(utterance_classes), intent_mapping


def default_intent_classification_parameters():
    return {
        "loss": 'log',
        "penalty": 'l2',
        "class_weight": 'balanced',
        "n_iter": 5,
        "random_state": 42,
        "n_jobs": -1
    }


class SnipsIntentClassifier(object):
    def __init__(self, language,
                 classifier_args=default_intent_classification_parameters()):
        self.language = language
        self.classifier_args = classifier_args
        self.classifier = None
        self.intent_list = None
        self.featurizer = Featurizer(self.language)
        self.min_utterances_per_intent = 20

    @property
    def fitted(self):
        return self.intent_list is not None

    def fit(self, dataset):
        utterances, y, intent_list = build_training_data(
            dataset, self.language,
            min_utterances_per_intent=self.min_utterances_per_intent)
        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self.featurizer = self.featurizer.fit(dataset, utterances, y)
        if self.featurizer is None:
            return self

        X = self.featurizer.transform(utterances)
        alpha = get_regularization_factor(dataset)
        self.classifier_args.update({'alpha': alpha})
        self.classifier = SGDClassifier(**self.classifier_args).fit(X, y)
        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError('SnipsIntentClassifier instance must be '
                                 'fitted before `get_intent` can be called')

        if len(text) == 0 or len(self.intent_list) == 0 \
                or self.featurizer is None or self.classifier is None:
            return None

        if len(self.intent_list) == 1:
            if self.intent_list[0] is None:
                return None
            return IntentClassificationResult(self.intent_list[0], 1.0)

        X = self.featurizer.transform([text])
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
            "classifier_args": self.classifier_args,
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": self.intent_list,
            "language_code": self.language.iso_code,
            "featurizer": featurizer_dict
        }

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict['language_code'])
        classifier_args = obj_dict['classifier_args']
        classifier = cls(language=language, classifier_args=classifier_args)
        sgd_classifier = None
        coeffs = obj_dict['coeffs']
        intercept = obj_dict['intercept']
        if coeffs is not None and intercept is not None:
            sgd_classifier = SGDClassifier(**classifier_args)
            sgd_classifier.coef_ = np.array(coeffs)
            sgd_classifier.intercept_ = np.array(intercept)
        classifier.classifier = sgd_classifier
        classifier.intent_list = obj_dict['intent_list']
        featurizer = obj_dict['featurizer']
        if featurizer is not None:
            classifier.featurizer = Featurizer.from_dict(featurizer)
        return classifier
