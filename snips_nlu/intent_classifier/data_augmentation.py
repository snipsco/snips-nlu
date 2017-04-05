import numpy as np

from intent_classifier_resources import get_subtitles
from snips_nlu.constants import INTENTS, UTTERANCES, UTTERANCE_TEXT
from snips_nlu.preprocessing import stem


def get_non_empty_intents(dataset):
    return [name for name, data in dataset[INTENTS].items() if
            len(data[UTTERANCES]) > 0]


def get_regularization_factor(dataset):
    intent_list = get_non_empty_intents(dataset)
    avg_utterances = np.mean([len(dataset[INTENTS][intent][UTTERANCES]) for
                             intent in intent_list])
    total_utterances = sum(len(dataset[INTENTS][intent][UTTERANCES]) for
                           intent in intent_list)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def augment_dataset(dataset, language, intent_list):
    noise_class = 0
    classes_mapping = {intent: i + 1 for i, intent in enumerate(intent_list)}
    avg_utterances = np.mean([len(dataset[INTENTS][intent][UTTERANCES]) for
                             intent in intent_list])

    noise = list(get_subtitles(language))
    noise_size = int(5 * avg_utterances)
    noisy_utterances = np.random.choice(noise, size=noise_size, replace=False)

    augmented_utterances = []
    utterance_classes = []
    for intent in intent_list:
        utterances = dataset[INTENTS][intent][UTTERANCES]
        augmented_utterances += [utterance[UTTERANCE_TEXT] for utterance in
                                 utterances]
        utterance_classes += [classes_mapping[intent] for _ in utterances]

    augmented_utterances += list(noisy_utterances)
    utterance_classes += [noise_class for _ in noisy_utterances]
    stemmed_utterances = [stem(utterance, language) for utterance in
                          augmented_utterances]

    return stemmed_utterances, np.array(utterance_classes)
