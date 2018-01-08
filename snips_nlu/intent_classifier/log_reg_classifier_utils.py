from __future__ import unicode_literals

import re
from copy import deepcopy
from uuid import uuid4

import itertools
import numpy as np

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (
    UNKNOWNWORD, INTENTS, UTTERANCES, DATA, ENTITY, TEXT)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.resources import get_noises
from snips_nlu.tokenization import tokenize_light

NOISE_NAME = str(uuid4()).decode()
WORD_REGEX = re.compile(r"\w+(\s+\w+)*")
UNKNOWNWORD_REGEX = re.compile(r"%s(\s+%s)*" % (UNKNOWNWORD, UNKNOWNWORD))


def remove_builtin_slots(dataset):
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


def get_noise_it(noise, mean_length, std_length, random_state):
    it = itertools.cycle(noise)
    while True:
        noise_length = int(random_state.normal(mean_length, std_length))
        yield " ".join(next(it) for _ in xrange(noise_length))


def generate_smart_noise(augmented_utterances, replacement_string, language):
    text_utterances = [get_text_from_chunks(u[DATA])
                       for u in augmented_utterances]
    vocab = [w for u in text_utterances for w in tokenize_light(u, language)]
    vocab = set(vocab)
    noise = get_noises(language)
    return [w if w in vocab else replacement_string for w in noise]


def generate_noise_utterances(augmented_utterances, num_intents,
                              data_augmentation_config, language,
                              random_state):
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
                            std_utterances_length, random_state)
    # Remove duplicate 'unknowword unknowword'
    return [UNKNOWNWORD_REGEX.sub(UNKNOWNWORD, next(noise_it))
            for _ in xrange(noise_size)]


def add_unknown_word_to_utterances(augmented_utterances, replacement_string,
                                   unknown_word_prob, random_state):
    for u in augmented_utterances:
        for chunk in u[DATA]:
            if ENTITY in chunk and not is_builtin_entity(chunk[ENTITY]) \
                    and random_state.rand() < unknown_word_prob:
                chunk[TEXT] = WORD_REGEX.sub(replacement_string, chunk[TEXT])
    return augmented_utterances


def build_training_data(dataset, language, data_augmentation_config,
                        random_state):
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
    for nb_utterance, intent_name in itertools.izip(nb_utterances,
                                                    intents.keys()):
        min_utterances_to_generate = max(
            data_augmentation_config.min_utterances, nb_utterance)
        utterances = augment_utterances(
            dataset, intent_name, language=language,
            min_utterances=min_utterances_to_generate,
            capitalization_ratio=0.0, random_state=random_state)
        augmented_utterances += utterances
        utterance_classes += [classes_mapping[intent_name] for _ in
                              xrange(len(utterances))]
    augmented_utterances = add_unknown_word_to_utterances(
        augmented_utterances,
        data_augmentation_config.unknown_words_replacement_string,
        data_augmentation_config.unknown_word_prob,
        random_state
    )

    # Adding noise
    noisy_utterances = generate_noise_utterances(
        augmented_utterances, len(intents), data_augmentation_config, language,
        random_state)
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
