from __future__ import unicode_literals

import random
from copy import deepcopy
from itertools import cycle

import numpy as np

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (UTTERANCES, DATA, ENTITY, TEXT, INTENTS,
                                 ENTITIES, CAPITALIZE)
from snips_nlu.resources import get_stop_words
from snips_nlu.tokenization import tokenize_light


def capitalize(text, language):
    tokens = tokenize_light(text, language)
    return language.default_sep.join(
        t.title() if t.lower() not in get_stop_words(language)
        else t.lower() for t in tokens)


def capitalize_utterances(utterances, entities, language, ratio):
    capitalized_utterances = []
    for utterance in utterances:
        capitalized_utterance = deepcopy(utterance)
        for i, chunk in enumerate(capitalized_utterance[DATA]):
            capitalized_utterance[DATA][i][TEXT] = chunk[TEXT].lower()
            if ENTITY not in chunk:
                continue
            entity_label = chunk[ENTITY]
            if is_builtin_entity(entity_label):
                continue
            if not entities[entity_label][CAPITALIZE]:
                continue
            if random.random() > ratio:
                continue
            capitalized_utterance[DATA][i][TEXT] = capitalize(
                chunk[TEXT], language)
        capitalized_utterances.append(capitalized_utterance)
    return capitalized_utterances


def generate_utterance(contexts_iterator, entities_iterators):
    context = deepcopy(next(contexts_iterator))
    context_data = []
    for i, chunk in enumerate(context[DATA]):
        if ENTITY in chunk:
            if not is_builtin_entity(chunk[ENTITY]):
                new_chunk = dict(chunk)
                new_chunk[TEXT] = deepcopy(
                    next(entities_iterators[new_chunk[ENTITY]]))
                context_data.append(new_chunk)
            else:
                context_data.append(chunk)
        else:
            context_data.append(chunk)
    context[DATA] = context_data
    return context


def get_contexts_iterator(dataset, intent_name):
    shuffled_utterances = np.random.permutation(
        dataset[INTENTS][intent_name][UTTERANCES])
    return cycle(shuffled_utterances)


def get_entities_iterators(dataset, intent_entities):
    entities_its = dict()
    for entity in intent_entities:
        shuffled_values = np.random.permutation(
            dataset[ENTITIES][entity][UTTERANCES].keys())
        entities_its[entity] = cycle(shuffled_values)
    return entities_its


def get_intent_entities(dataset, intent_name):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent_name][UTTERANCES]:
        for chunk in utterance[DATA]:
            if ENTITY in chunk:
                intent_entities.add(chunk[ENTITY])
    return intent_entities


def num_queries_to_generate(dataset, intent_name, min_utterances):
    nb_utterances = len(dataset[INTENTS][intent_name][UTTERANCES])
    return max(nb_utterances, min_utterances)


def augment_utterances(dataset, intent_name, language, min_utterances,
                       capitalization_ratio):
    contexts_it = get_contexts_iterator(dataset, intent_name)
    intent_entities = get_intent_entities(dataset, intent_name)
    intent_entities = [e for e in intent_entities if not is_builtin_entity(e)]
    entities_its = get_entities_iterators(dataset, intent_entities)
    generated_utterances = []
    nb_to_generate = num_queries_to_generate(dataset, intent_name,
                                             min_utterances)
    while nb_to_generate > 0:
        generated_utterance = generate_utterance(contexts_it, entities_its)
        generated_utterances.append(generated_utterance)
        nb_to_generate -= 1

    generated_utterances = capitalize_utterances(
        generated_utterances, dataset[ENTITIES], language,
        ratio=capitalization_ratio)

    return generated_utterances
