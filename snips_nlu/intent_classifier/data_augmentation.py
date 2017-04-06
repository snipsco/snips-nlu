import numpy as np

from snips_nlu.intent_classifier.intent_classifier_resources import \
    get_subtitles
from snips_nlu.constants import INTENTS, UTTERANCES, UTTERANCE_TEXT
from snips_nlu.preprocessing import stem


def get_regularization_factor(dataset):
    intents = dataset[INTENTS]
    nb_utterances = [len(intent[UTTERANCES]) for intent in intents.values()]
    avg_utterances = np.mean(nb_utterances)
    total_utterances = sum(nb_utterances)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def build_training_data(custom_dataset, builtin_dataset, language,
                        noise_factor=5, use_stemming=True):
    # Separating custom intents from builtin intents
    custom_intents = custom_dataset[INTENTS]
    builtin_intents = builtin_dataset[INTENTS]
    all_intents = set(custom_intents.keys() + builtin_intents.keys())

    # Creating class mapping
    noise_class = 0
    classes_mapping = {intent: i + 1 for i, intent in
                       enumerate(custom_intents)}
    classes_mapping.update({intent: noise_class for intent in builtin_intents})

    # Computing dataset statistics
    nb_utterances = [len(intent[UTTERANCES]) for intent in
                     custom_intents.values()]
    avg_utterances = np.mean(nb_utterances)
    max_utterances = max(nb_utterances)

    # Adding custom and builtin utterances
    augmented_utterances = []
    utterance_classes = []
    for intent in all_intents:
        if intent in builtin_intents:
            utterances = builtin_dataset[INTENTS][intent][UTTERANCES][
                         :max_utterances]
        else:
            utterances = custom_dataset[INTENTS][intent][UTTERANCES]
        augmented_utterances += [utterance[UTTERANCE_TEXT] for utterance in
                                 utterances]
        utterance_classes += [classes_mapping[intent] for _ in utterances]

    # Adding noise
    noise = list(get_subtitles(language))
    noise_size = min(int(noise_factor * avg_utterances), len(noise))
    noisy_utterances = np.random.choice(noise, size=noise_size, replace=False)
    augmented_utterances += list(noisy_utterances)
    utterance_classes += [noise_class for _ in noisy_utterances]

    # Stemming utterances
    if use_stemming:
        augmented_utterances = [stem(utterance, language) for utterance in
                                augmented_utterances]

    return augmented_utterances, np.array(utterance_classes)
