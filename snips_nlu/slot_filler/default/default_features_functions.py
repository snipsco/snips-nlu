from __future__ import unicode_literals

import numpy as np
from nlu_utils import normalize

from snips_nlu.builtin_entities import _SUPPORTED_BUILTINS_BY_LANGUAGE, \
    is_builtin_entity
from snips_nlu.constants import UTTERANCES, DATA, ENTITY, ENTITIES, INTENTS
from snips_nlu.data_augmentation import get_contexts_iterator, \
    num_queries_to_generate
from snips_nlu.preprocessing import stem
from snips_nlu.slot_filler.crf_utils import TaggingScheme


def get_intent_custom_entities(dataset, intent):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent][UTTERANCES]:
        for c in utterance[DATA]:
            if ENTITY in c:
                intent_entities.add(c[ENTITY])
    custom_entities = dict()
    for ent in intent_entities:
        if not is_builtin_entity(ent):
            custom_entities[ent] = dataset[ENTITIES][ent]
    return custom_entities


def get_num_entity_appearances(dataset, intent, entity, config):
    contexts_it = get_contexts_iterator(dataset, intent)
    nb_to_generate = num_queries_to_generate(
        dataset, intent, config.data_augmentation_config.min_utterances)
    contexts = [next(contexts_it)[DATA] for _ in xrange(nb_to_generate)]
    return sum(1 for q in contexts for c in q
               if ENTITY in c and c[ENTITY] == entity)


def compute_entity_collection_size(dataset, entity, config):
    num_entities = len(dataset[ENTITIES][entity][UTTERANCES])
    collection_size = int(config.crf_features_config.base_drop_ratio
                          * num_entities)
    collection_size = max(collection_size, 1)
    collection_size = min(collection_size, num_entities)
    return collection_size


def default_features(language, dataset, intent, config, use_stemming,
                     common_words_gazetteer_name=None):
    features = [
        {
            "factory_name": "get_ngram_fn",
            "args": {
                "n": 1,
                "use_stemming": use_stemming,
                "language_code": language.iso_code,
                "common_words_gazetteer_name": common_words_gazetteer_name
            },
            "offsets": [-2, -1, 0, 1, 2]
        },
        {
            "factory_name": "get_ngram_fn",
            "args": {
                "n": 2,
                "use_stemming": use_stemming,
                "language_code": language.iso_code,
                "common_words_gazetteer_name": common_words_gazetteer_name
            },
            "offsets": [-2, 1]
        },
        {
            "factory_name": "is_digit",
            "args": {},
            "offsets": [-1, 0, 1]
        },
        {
            "factory_name": "is_first",
            "args": {},
            "offsets": [-2, -1, 0]
        },
        {
            "factory_name": "is_last",
            "args": {},
            "offsets": [0, 1, 2]
        }
    ]

    # Built-ins
    for entity in _SUPPORTED_BUILTINS_BY_LANGUAGE[language]:
        features.append(
            {
                "factory_name": "get_built_in_annotation_fn",
                "args": {
                    "built_in_entity_label": entity.label,
                    "language_code": language.iso_code,
                    "tagging_scheme_code": TaggingScheme.BILOU.value},
                "offsets": [-2, -1, 0]
            }
        )

    # Entity lookup
    def preprocess(string):
        normalized = normalize(string)
        return stem(normalized, language) if use_stemming else normalized

    intent_entities = get_intent_custom_entities(dataset, intent)
    for entity_name, entity in intent_entities.iteritems():
        if len(entity[UTTERANCES]) == 0:
            continue

        collection = list(
            set(preprocess(e) for e in entity[UTTERANCES].keys()))
        collection_size = compute_entity_collection_size(dataset, entity_name,
                                                         config)
        collection = np.random.choice(collection, collection_size,
                                      replace=False).tolist()
        features.append(
            {
                "factory_name": "get_token_is_in_fn",
                "args": {"tokens_collection": collection,
                         "collection_name": entity_name,
                         "use_stemming": use_stemming,
                         "language_code": language.iso_code,
                         "tagging_scheme_code": TaggingScheme.BILOU.value},
                "offsets": tuple(config.crf_features_config.entities_offsets)
            }
        )
    return features


def default_shape_ngram_features(language):
    return [
        {
            "factory_name": "get_shape_ngram_fn",
            "args": {"n": 1, "language_code": language.iso_code},
            "offsets": [0]
        },
        {
            "factory_name": "get_shape_ngram_fn",
            "args": {"n": 2, "language_code": language.iso_code},
            "offsets": [-1, 0]
        },
        {
            "factory_name": "get_shape_ngram_fn",
            "args": {"n": 3, "language_code": language.iso_code},
            "offsets": [-1]
        }
    ]
