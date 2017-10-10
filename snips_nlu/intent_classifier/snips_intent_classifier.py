from __future__ import unicode_literals

from itertools import izip, cycle
from uuid import uuid4

import numpy as np
from sklearn.linear_model import SGDClassifier

from feature_extraction import Featurizer
from snips_nlu.config import IntentClassifierConfig
from snips_nlu.constants import INTENTS, UTTERANCES, DATA
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.languages import Language
from snips_nlu.resources import get_noises
from snips_nlu.result import IntentClassificationResult
from snips_nlu.tokenization import tokenize_light

NOISE_NAME = str(uuid4()).decode()


def get_regularization_factor(dataset):
    intents = dataset[INTENTS]
    nb_utterances = [len(intent[UTTERANCES]) for intent in intents.values()]
    avg_utterances = np.mean(nb_utterances)
    total_utterances = sum(nb_utterances)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def get_noise_it(noise, mean_length, std_length, language):
    it = cycle(noise)
    while True:
        noise_length = int(np.random.normal(mean_length, std_length))
        yield " ".join(next(it) for _ in xrange(noise_length))


def generate_noise_utterances(augmented_utterances, num_intents, config,
                              language):
    if not len(augmented_utterances) or not num_intents:
        return []
    avg_num_utterances = len(augmented_utterances) / float(num_intents)
    noise = get_noises(language)
    noise_size = min(int(config.noise_factor * avg_num_utterances), len(noise))
    utterances_lengths = [len(tokenize_light(u, language))
                          for u in augmented_utterances]
    mean_utterances_length = np.mean(utterances_lengths)
    std_utterances_length = np.std(utterances_lengths)
    noise_it = get_noise_it(noise, mean_utterances_length,
                            std_utterances_length, language)
    return [next(noise_it) for _ in xrange(noise_size)]


def build_training_data(dataset, language, config):
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
        min_utterances_to_generate = max(
            config.data_augmentation_config.min_utterances, nb_utterance)
        utterances = augment_utterances(
            dataset, intent_name, language=language,
            min_utterances=min_utterances_to_generate,
            capitalization_ratio=config.data_augmentation_config
                .capitalization_ratio)
        augmented_utterances += [get_text_from_chunks(utterance[DATA]) for
                                 utterance in utterances]
        utterance_classes += [classes_mapping[intent_name] for _ in utterances]

    # Adding noise
    noisy_utterances = generate_noise_utterances(
        augmented_utterances, len(intents), config, language)

    augmented_utterances += noisy_utterances
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


class SnipsIntentClassifier(object):
    def __init__(self, language, config=IntentClassifierConfig()):
        self.language = language
        self.config = config
        self.classifier = None
        self.intent_list = None
        self.featurizer = Featurizer(self.language)
        self.min_utterances_per_intent = 20

    @property
    def fitted(self):
        return self.intent_list is not None

    def fit(self, dataset):
        utterances, y, intent_list = build_training_data(
            dataset, self.language, self.config)
        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self.featurizer = self.featurizer.fit(dataset, utterances, y)
        if self.featurizer is None:
            return self

        X = self.featurizer.transform(utterances)
        alpha = get_regularization_factor(dataset)
        self.config.log_reg_args['alpha'] = alpha
        self.classifier = SGDClassifier(**self.config.log_reg_args).fit(X, y)
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
            "config": self.config.to_dict(),
            "coeffs": coeffs,
            "intercept": intercept,
            "intent_list": self.intent_list,
            "language_code": self.language.iso_code,
            "featurizer": featurizer_dict
        }

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict['language_code'])
        config = IntentClassifierConfig.from_dict(obj_dict["config"])
        classifier = cls(language=language, config=config)
        sgd_classifier = None
        coeffs = obj_dict['coeffs']
        intercept = obj_dict['intercept']
        if coeffs is not None and intercept is not None:
            sgd_classifier = SGDClassifier(**config.log_reg_args)
            sgd_classifier.coef_ = np.array(coeffs)
            sgd_classifier.intercept_ = np.array(intercept)
        classifier.classifier = sgd_classifier
        classifier.intent_list = obj_dict['intent_list']
        featurizer = obj_dict['featurizer']
        if featurizer is not None:
            classifier.featurizer = Featurizer.from_dict(featurizer)
        return classifier
