import numpy as np

from intent_classifier_resources import get_subtitles


def check_if_empty(dataset):
    intent_list = dataset['intents'].keys()

    queries_per_intent = [len(dataset['intents'][intent]['utterances']) for intent in intent_list]
    ind_0 = [i for i, x in enumerate(queries_per_intent) if x == 0]

    intent_list = [intent for i, intent in enumerate(intent_list) if i not in ind_0]

    return ['None'] + intent_list, len(intent_list) == 0

def get_non_empty_intents(dataset):
    return [name for name, data in dataset["intents"].items() if len(data["utterances"]) > 0]


def augment_dataset(dataset, intent_list, language='en'):
    intent_code = dict((intent, i) for i, intent in enumerate(intent_list))

    queries_per_intent = [len(dataset['intents'][intent]['utterances']) for intent in intent_list]
    mean_queries_per_intent = np.mean(queries_per_intent)

    alpha = 1.0 / (4 * (sum(queries_per_intent) + 5 * mean_queries_per_intent))

    data_noise_train = list(get_subtitles(language))
    queries_noise = np.random.choice(data_noise_train, size=int(5 * mean_queries_per_intent), replace=False)

    queries = []
    y = []
    for intent in intent_list:
        utterances = dataset['intents'][intent]['utterances']
        queries += [''.join([utterances[i]['data'][j]['text'] for j in xrange(len(utterances[i]['data']))]) for i in
                    xrange(len(utterances))]
        y += [intent_code[intent] for _ in xrange(len(dataset['intents'][intent]['utterances']))]

    queries += [query for query in queries_noise]
    y += [0 for _ in xrange(len(queries_noise))]

    queries = np.array(queries)
    y = np.array(y)

    return (queries, y), alpha
