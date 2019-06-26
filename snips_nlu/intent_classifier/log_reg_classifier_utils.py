from __future__ import division, unicode_literals

import itertools
import re
from builtins import next, range, str, zip
from copy import deepcopy
from uuid import uuid4

import numpy as np
from future.utils import iteritems, itervalues

from snips_nlu.common.utils import check_random_state
from snips_nlu.constants import (DATA, ENTITY, INTENTS, TEXT,
                                 UNKNOWNWORD, UTTERANCES)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.resources import get_bigrams, get_noise

NOISE_NAME = str(uuid4())
WORD_REGEX = re.compile(r"\w+(\s+\w+)*")
UNKNOWNWORD_REGEX = re.compile(r"%s(\s+%s)*" % (UNKNOWNWORD, UNKNOWNWORD))


def remove_builtin_slots(dataset):
    filtered_dataset = deepcopy(dataset)
    for intent_data in itervalues(filtered_dataset[INTENTS]):
        for utterance in intent_data[UTTERANCES]:
            utterance[DATA] = [
                chunk for chunk in utterance[DATA]
                if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])]
    return filtered_dataset


def get_regularization_factor(dataset):
    intents = dataset[INTENTS]
    nb_utterances = [len(intent[UTTERANCES]) for intent in itervalues(intents)]
    avg_utterances = np.mean(nb_utterances)
    total_utterances = sum(nb_utterances)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def get_noise_it(noise, mean_length, std_length, random_state):
    it = itertools.cycle(noise)
    while True:
        noise_length = int(random_state.normal(mean_length, std_length))
        # pylint: disable=stop-iteration-return
        yield " ".join(next(it) for _ in range(noise_length))
        # pylint: enable=stop-iteration-return


def get_bigrams_it(bigrams, bigrams_frequencies, random_state):
    random_state = check_random_state(random_state)
    bigrams_indices = range(len(bigrams))
    while True:
        bigram_idx = random_state.choice(bigrams_indices,
                                         p=bigrams_frequencies)
        yield bigrams[bigram_idx]


def get_lm_unigram_it(bigrams, bigrams_frequencies, random_state):
    previous_unigram = None
    while True:
        random_state = check_random_state(random_state)
        bigrams_indices = range(len(bigrams))
        if previous_unigram is None:
            bigram_idx = random_state.choice(bigrams_indices,
                                             p=bigrams_frequencies)
            unigram = bigrams[bigram_idx][0]
        else:
            filtered_bigrams = [
                (bigram, freq)
                for bigram, freq in zip(bigrams, bigrams_frequencies)
                if bigram[0] == previous_unigram]
            if not filtered_bigrams:
                bigram_idx = random_state.choice(bigrams_indices,
                                                 p=bigrams_frequencies)
                unigram = bigrams[bigram_idx][0]
            else:
                filtered_frequencies = [b[1] for b in filtered_bigrams]
                norm = np.linalg.norm(filtered_frequencies, 1)
                filtered_frequencies = filtered_frequencies / norm
                filtered_bigrams = [b[0] for b in filtered_bigrams]
                filtered_bigrams_indices = range(len(filtered_bigrams))
                filtered_bigram_idx = random_state.choice(
                    filtered_bigrams_indices, p=filtered_frequencies)
                unigram = filtered_bigrams[filtered_bigram_idx][1]

        previous_unigram = unigram
        yield unigram


def get_bigrams_noise_it(
        bigrams, bigrams_frequencies, mean_length, std_length, random_state):
    bigrams_it = get_bigrams_it(bigrams, bigrams_frequencies, random_state)
    while True:
        noise_length = int(random_state.normal(mean_length, std_length))
        nb_bigrams = int(max(noise_length / 2, 1))
        yield " ".join(w for _ in range(nb_bigrams) for w in next(bigrams_it))


def get_lm_noise_it(bigrams, bigrams_frequencies, mean_length, std_length,
                    random_state):
    while True:
        it = get_lm_unigram_it(bigrams, bigrams_frequencies, random_state)
        noise_length = int(random_state.normal(mean_length, std_length))
        yield " ".join(next(it) for _ in range(noise_length))


def generate_smart_noise(noise, augmented_utterances, replacement_string,
                         language):
    text_utterances = [get_text_from_chunks(u[DATA])
                       for u in augmented_utterances]
    vocab = [w for u in text_utterances for w in tokenize_light(u, language)]
    vocab = set(vocab)
    return [w if w in vocab else replacement_string for w in noise]


def generate_noise_utterances(augmented_utterances, resources, num_intents,
                              data_augmentation_config, language,
                              random_state):
    if not augmented_utterances or not num_intents:
        return []

    avg_num_utterances = len(augmented_utterances) / float(num_intents)
    noise_size = int(
        data_augmentation_config.noise_factor * avg_num_utterances)
    utterances_lengths = [
        len(tokenize_light(get_text_from_chunks(u[DATA]), language))
        for u in augmented_utterances]
    mean_utterances_length = np.mean(utterances_lengths)
    std_utterances_length = np.std(utterances_lengths)
    if data_augmentation_config.use_bigrams_noise:
        bigrams = get_bigrams(resources)
        noise_it = get_bigrams_noise_it(
            bigrams["bigrams"], bigrams["frequencies"], mean_utterances_length,
            std_utterances_length, random_state)
        # noise_it = get_lm_noise_it(
        #     bigrams["bigrams"], bigrams["frequencies"], mean_utterances_length,
        #     std_utterances_length, random_state)
    else:
        noise = get_noise(resources)
        if data_augmentation_config.unknown_words_replacement_string \
                is not None:
            noise = generate_smart_noise(
                noise, augmented_utterances,
                data_augmentation_config.unknown_words_replacement_string,
                language)
        noise_it = get_noise_it(noise, mean_utterances_length,
                                std_utterances_length, random_state)

    # Remove duplicate 'unknownword unknownword'
    return [
        text_to_utterance(UNKNOWNWORD_REGEX.sub(UNKNOWNWORD, next(noise_it)))
        for _ in range(noise_size)]


def add_unknown_word_to_utterances(utterances, replacement_string,
                                   unknown_word_prob, max_unknown_words,
                                   random_state):
    if not max_unknown_words:
        return utterances

    new_utterances = deepcopy(utterances)
    for u in new_utterances:
        if random_state.rand() < unknown_word_prob:
            num_unknown = random_state.randint(1, max_unknown_words + 1)
            # We choose to put the noise at the end of the sentence and not
            # in the middle so that it doesn't impact to much ngrams
            # computation
            extra_chunk = {
                TEXT: " " + " ".join(
                    replacement_string for _ in range(num_unknown))
            }
            u[DATA].append(extra_chunk)
    return new_utterances


def build_training_data(dataset, language, data_augmentation_config, resources,
                        random_state):
    # Create class mapping
    intents = dataset[INTENTS]
    intent_index = 0
    classes_mapping = dict()
    for intent in sorted(intents):
        classes_mapping[intent] = intent_index
        intent_index += 1

    noise_class = intent_index

    # Computing dataset statistics
    nb_utterances = [len(intent[UTTERANCES]) for intent in itervalues(intents)]

    augmented_utterances = []
    utterance_classes = []
    for nb_utterance, intent_name in zip(nb_utterances, intents):
        min_utterances_to_generate = max(
            data_augmentation_config.min_utterances, nb_utterance)
        utterances = augment_utterances(
            dataset, intent_name, language=language,
            min_utterances=min_utterances_to_generate,
            capitalization_ratio=0.0,
            add_builtin_entities_examples=
            data_augmentation_config.add_builtin_entities_examples,
            resources=resources, random_state=random_state)
        augmented_utterances += utterances
        utterance_classes += [classes_mapping[intent_name] for _ in
                              range(len(utterances))]
    if data_augmentation_config.unknown_words_replacement_string is not None:
        augmented_utterances = add_unknown_word_to_utterances(
            augmented_utterances,
            data_augmentation_config.unknown_words_replacement_string,
            data_augmentation_config.unknown_word_prob,
            data_augmentation_config.max_unknown_words,
            random_state
        )

    # Adding noise
    noisy_utterances = generate_noise_utterances(
        augmented_utterances, resources, len(intents),
        data_augmentation_config,
        language, random_state)

    augmented_utterances += noisy_utterances
    utterance_classes += [noise_class for _ in noisy_utterances]
    if noisy_utterances:
        classes_mapping[NOISE_NAME] = noise_class

    nb_classes = len(set(itervalues(classes_mapping)))
    intent_mapping = [None for _ in range(nb_classes)]
    for intent, intent_class in iteritems(classes_mapping):
        if intent == NOISE_NAME:
            intent_mapping[intent_class] = None
        else:
            intent_mapping[intent_class] = intent

    return augmented_utterances, np.array(utterance_classes), intent_mapping


def text_to_utterance(text):
    return {DATA: [{TEXT: text}]}
