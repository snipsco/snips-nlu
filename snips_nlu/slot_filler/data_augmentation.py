from itertools import cycle

import numpy as np

from snips_nlu.constants import (UTTERANCES, DATA, ENTITY, USE_SYNONYMS,
                                 SYNONYMS, VALUE, TEXT, INTENTS, ENTITIES)


def generate_utterance(contexts_iterator, entities_iterators):
    context = next(contexts_iterator)
    for i, chunk in enumerate(context[DATA]):
        if ENTITY in chunk:
            new_chunk = dict(chunk)
            new_chunk[TEXT] = next(entities_iterators[new_chunk[ENTITY]])
            context[DATA][i] = new_chunk
    return context


def get_contexts_iterator(intent_utterances):
    shuffled_utterances = np.random.permutation(intent_utterances)
    return cycle(shuffled_utterances)


def get_entities_iterators(dataset, intent_entities):
    entities_its = dict()
    for entity in intent_entities:
        if dataset[ENTITIES][entity][USE_SYNONYMS]:
            values = [s for d in dataset[ENTITIES][entity][DATA] for s in
                      d[SYNONYMS]]
        else:
            values = [d[VALUE] for d in dataset[ENTITIES][entity][DATA]]
        shuffled_values = np.random.permutation(values)
        entities_its[entity] = cycle(shuffled_values)
    return entities_its


def get_intent_entities(dataset, intent_name):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent_name][UTTERANCES]:
        for chunk in utterance[DATA]:
            if ENTITY in chunk:
                intent_entities.add(chunk[ENTITY])
    return intent_entities


def augment_utterances(dataset, intent_name, max_utterances):
    utterances = dataset[INTENTS][intent_name][UTTERANCES]
    if max_utterances < len(utterances):
        return utterances

    num_to_generate = max_utterances - len(utterances)
    contexts_it = get_contexts_iterator(utterances)
    intent_entities = get_intent_entities(dataset, intent_name)
    entities_its = get_entities_iterators(dataset, intent_entities)
    while num_to_generate > 0:
        utterances.append(generate_utterance(contexts_it, entities_its))
        num_to_generate -= 1

    return utterances
