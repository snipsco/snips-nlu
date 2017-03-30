import numpy as np

from intent_classifier_resources import get_subtitles
from snips_nlu.constants import TEXT, DATA, INTENTS, UTTERANCES, LANGUAGE


def get_non_empty_intents(dataset):
    return [name for name, data in dataset[INTENTS].items() if
            len(data[UTTERANCES]) > 0]


def augment_dataset(dataset, language, intent_list):
    intent_code = dict((intent, i + 1) for i, intent in enumerate(intent_list))

    queries_per_intent = [len(dataset[INTENTS][intent][UTTERANCES]) for
                          intent in intent_list]
    mean_queries_per_intent = np.mean(queries_per_intent)

    alpha = 1.0 / (4 * (sum(queries_per_intent) + 5 * mean_queries_per_intent))

    data_noise_train = list(get_subtitles(language))
    queries_noise = np.random.choice(data_noise_train,
                                     size=int(5 * mean_queries_per_intent),
                                     replace=False)

    queries = []
    y = []
    for intent in intent_list:
        utterances = dataset[INTENTS][intent][UTTERANCES]
        queries += [''.join([utterances[i][DATA][j][TEXT] for j in
                             xrange(len(utterances[i][DATA]))]) for i in
                    xrange(len(utterances))]
        y += [intent_code[intent] for _ in xrange(len(utterances))]

    queries += [query for query in queries_noise]
    y += [0 for _ in xrange(len(queries_noise))]

    queries = np.array(queries)
    y = np.array(y)

    return (queries, y), alpha
