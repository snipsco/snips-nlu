from uuid import uuid4

import numpy as np

from snips_nlu.constants import INTENTS, UTTERANCES, DATA
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.preprocessing import stem_sentence
from snips_nlu.resources import get_subtitles

NOISE_NAME = str(uuid4()).decode()


def get_regularization_factor(dataset):
    intents = dataset[INTENTS]
    nb_utterances = [len(intent[UTTERANCES]) for intent in intents.values()]
    avg_utterances = np.mean(nb_utterances)
    total_utterances = sum(nb_utterances)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def build_training_data(dataset, language,
                        noise_factor=5, use_stemming=True):
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
    for intent_name, intent in intents.iteritems():
        utterances = intent[UTTERANCES]
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

    # Stemming utterances
    if use_stemming:
        augmented_utterances = [stem_sentence(utterance, language) for
                                utterance in augmented_utterances]

    nb_classes = len(set(classes_mapping.values()))
    intent_mapping = [None for _ in xrange(nb_classes)]
    for intent, intent_class in classes_mapping.iteritems():
        if intent == NOISE_NAME:
            intent_mapping[intent_class] = None
        else:
            intent_mapping[intent_class] = intent

    return augmented_utterances, np.array(utterance_classes), intent_mapping
