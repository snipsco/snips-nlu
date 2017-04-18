import numpy as np
from duckling import core

from snips_nlu.built_in_entities import BuiltInEntity
from snips_nlu.constants import DATA, USE_SYNONYMS, SYNONYMS, VALUE
from snips_nlu.preprocessing import stem
from snips_nlu.slot_filler.crf_utils import TaggingScheme


def default_features(module_name, language, intent_entities, use_stemming,
                     entities_offsets, entity_keep_prob,
                     common_words_gazetteer_name=None):
    features = [
        {
            "module_name": module_name,
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
            "module_name": module_name,
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
            "module_name": module_name,
            "factory_name": "get_shape_ngram_fn",
            "args": {"n": 1},
            "offsets": [0]
        },
        {
            "module_name": module_name,
            "factory_name": "get_shape_ngram_fn",
            "args": {"n": 2},
            "offsets": [-1, 0]
        },
        {
            "module_name": module_name,
            "factory_name": "get_shape_ngram_fn",
            "args": {"n": 3},
            "offsets": [-1]
        },
        {
            "module_name": module_name,
            "factory_name": "is_digit",
            "args": {},
            "offsets": [-1, 0, 1]
        },
        {
            "module_name": module_name,
            "factory_name": "is_first",
            "args": {},
            "offsets": [-2, -1, 0]
        },
        {
            "module_name": module_name,
            "factory_name": "is_last",
            "args": {},
            "offsets": [0, 1, 2]
        }
    ]

    # Built-ins
    for dim in core.get_dims(language.iso_code):
        if dim not in BuiltInEntity.built_in_entity_by_duckling_dim:
            continue
        entity = BuiltInEntity.from_duckling_dim(dim)
        features.append(
            {
                "module_name": module_name,
                "factory_name": "get_built_in_annotation_fn",
                "args": {
                    "built_in_entity_label": entity.label,
                    "language_code": language.iso_code,
                    "tagging_scheme_code": TaggingScheme.BILOU.value},
                "offsets": [-2, -1, 0]
            }
        )

    # Entity lookup
    if use_stemming:
        preprocess = lambda string: stem(string, language)
    else:
        preprocess = lambda string: string

    for entity_name, entity in intent_entities.iteritems():
        if len(entity[DATA]) == 0:
            continue
        if entity[USE_SYNONYMS]:
            collection = [preprocess(s) for d in entity[DATA] for s in
                          d[SYNONYMS]]
        else:
            collection = [preprocess(d[VALUE]) for d in entity[DATA]]
        collection_size = max(int(entity_keep_prob * len(collection)), 1)
        collection = np.random.choice(collection, collection_size,
                                      replace=False).tolist()
        features.append(
            {
                "module_name": module_name,
                "factory_name": "get_token_is_in_fn",
                "args": {"tokens_collection": collection,
                         "collection_name": entity_name,
                         "use_stemming": use_stemming,
                         "tagging_scheme_code": TaggingScheme.BILOU.value},
                "offsets": entities_offsets
            }
        )
    return features
