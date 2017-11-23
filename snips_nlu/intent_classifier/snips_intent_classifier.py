from __future__ import unicode_literals

import re
from itertools import izip, cycle
from random import random
from uuid import uuid4

import numpy as np
from copy import deepcopy
from sklearn.linear_model import SGDClassifier

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.config import IntentClassifierConfig
from snips_nlu.constants import (INTENTS, UTTERANCES, DATA, ENTITY,
                                 UNKNOWNWORD, TEXT)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.intent_classifier.feature_extraction import Featurizer
from snips_nlu.languages import Language
from snips_nlu.resources import get_noises
from snips_nlu.result import IntentClassificationResult
from snips_nlu.tokenization import tokenize_light

NOISE_NAME = str(uuid4()).decode()

WORD_REGEX = re.compile(r"\w+(\s+\w+)*")
UNKNOWNWORD_REGEX = re.compile(r"%s(\s+%s)*" % (UNKNOWNWORD, UNKNOWNWORD))


def remove_builtin_slot(dataset):
    filtered_dataset = deepcopy(dataset)
    for intent_data in filtered_dataset[INTENTS].values():
        for utterance in intent_data[UTTERANCES]:
            utterance[DATA] = [
                chunk for chunk in utterance[DATA]
                if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])]
    return filtered_dataset


def get_regularization_factor(dataset):
    intents = dataset[INTENTS]
    nb_utterances = [len(intent[UTTERANCES]) for intent in intents.values()]
    avg_utterances = np.mean(nb_utterances)
    total_utterances = sum(nb_utterances)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def get_noise_it(noise, mean_length, std_length):
    it = cycle(noise)
    while True:
        noise_length = int(np.random.normal(mean_length, std_length))
        yield " ".join(next(it) for _ in xrange(noise_length))


def generate_smart_noise(augmented_utterances, replacement_string, language):
    text_utterances = [get_text_from_chunks(u[DATA])
                       for u in augmented_utterances]
    vocab = [w for u in text_utterances for w in tokenize_light(u, language)]
    vocab = set(vocab)
    noise = get_noises(language)
    return [w if w in vocab else replacement_string for w in noise]


def generate_noise_utterances(augmented_utterances, num_intents,
                              data_augmentation_config, language):
    if not augmented_utterances or not num_intents:
        return []
    avg_num_utterances = len(augmented_utterances) / float(num_intents)
    if data_augmentation_config.unknown_words_replacement_string is not None:
        noise = generate_smart_noise(
            augmented_utterances,
            data_augmentation_config.unknown_words_replacement_string,
            language)
    else:
        noise = get_noises(language)

    noise_size = min(
        int(data_augmentation_config.noise_factor * avg_num_utterances),
        len(noise))
    utterances_lengths = [
        len(tokenize_light(get_text_from_chunks(u[DATA]), language))
        for u in augmented_utterances]
    mean_utterances_length = np.mean(utterances_lengths)
    std_utterances_length = np.std(utterances_lengths)
    noise_it = get_noise_it(noise, mean_utterances_length,
                            std_utterances_length)
    # Remove duplicate 'unknowword unknowword'
    return [UNKNOWNWORD_REGEX.sub(UNKNOWNWORD, next(noise_it))
            for _ in xrange(noise_size)]


def add_unknown_word_to_utterances(augmented_utterances, replacement_string,
                                   unknown_word_prob):
    for u in augmented_utterances:
        for chunk in u[DATA]:
            if ENTITY in chunk and not is_builtin_entity(chunk[ENTITY]) \
                    and random() < unknown_word_prob:
                chunk[TEXT] = WORD_REGEX.sub(replacement_string, chunk[TEXT])
    return augmented_utterances


def build_training_data(dataset, language, data_augmentation_config):
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
            data_augmentation_config.min_utterances, nb_utterance)
        utterances = augment_utterances(
            dataset, intent_name, language=language,
            min_utterances=min_utterances_to_generate,
            capitalization_ratio=0.0)  # Data is anyway lower with `normalize`
        augmented_utterances += utterances
        utterance_classes += [classes_mapping[intent_name] for _ in
                              xrange(len(utterances))]
    augmented_utterances = add_unknown_word_to_utterances(
        augmented_utterances,
        data_augmentation_config.unknown_words_replacement_string,
        data_augmentation_config.unknown_word_prob
    )

    # Adding noise
    noisy_utterances = generate_noise_utterances(
        augmented_utterances, len(intents),
        data_augmentation_config, language)
    augmented_utterances = [get_text_from_chunks(u[DATA])
                            for u in augmented_utterances]

    augmented_utterances += noisy_utterances
    utterance_classes += [noise_class for _ in noisy_utterances]
    if noisy_utterances:
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
        self.featurizer = Featurizer(
            self.language,
            self.config.data_augmentation_config
                .unknown_words_replacement_string,
            self.config.featurizer_config)
        self.min_utterances_per_intent = 20

    @property
    def fitted(self):
        return self.intent_list is not None

    def fit(self, dataset):
        filtered_dataset = remove_builtin_slot(dataset)
        utterances, y, intent_list = build_training_data(
            filtered_dataset, self.language,
            self.config.data_augmentation_config)
        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self.featurizer = self.featurizer.fit(filtered_dataset, utterances, y)
        if self.featurizer is None:
            return self

        X = self.featurizer.transform(utterances)  # pylint: disable=C0103
        alpha = get_regularization_factor(filtered_dataset)
        self.config.log_reg_args['alpha'] = alpha
        self.classifier = SGDClassifier(**self.config.log_reg_args).fit(X, y)
        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError('SnipsIntentClassifier instance must be '
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
