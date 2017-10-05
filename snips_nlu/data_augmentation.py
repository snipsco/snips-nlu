from __future__ import unicode_literals

from copy import deepcopy
from itertools import cycle

import numpy as np

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (UTTERANCES, DATA, ENTITY, TEXT, INTENTS,
                                 ENTITIES)
from snips_nlu.utils import namedtuple_with_defaults

_DataAugmentationConfig = namedtuple_with_defaults(
    '_DataAugmentationConfig',
    'max_utterances',
    {
        'max_utterances': 200
    }
)


class DataAugmentationConfig(_DataAugmentationConfig):
    def to_dict(self):
        return self._asdict()

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


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


def get_contexts_iterator(intent_utterances):
    shuffled_utterances = np.random.permutation(intent_utterances)
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


def augment_utterances(dataset, intent_name, language, max_utterances):
    utterances = dataset[INTENTS][intent_name][UTTERANCES]
    nb_utterances = len(utterances)
    nb_to_generate = max(nb_utterances, max_utterances)
    contexts_it = get_contexts_iterator(utterances)
    intent_entities = get_intent_entities(dataset, intent_name)
    intent_entities = [e for e in intent_entities if not is_builtin_entity(e)]
    entities_its = get_entities_iterators(dataset, intent_entities)
    generated_utterances = []
    while nb_to_generate > 0:
        generated_utterance = generate_utterance(contexts_it, entities_its)
        generated_utterances.append(generated_utterance)
        nb_to_generate -= 1

    return generated_utterances
