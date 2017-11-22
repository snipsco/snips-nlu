from __future__ import unicode_literals

from nlu_utils import normalize

from snips_nlu.builtin_entities import _SUPPORTED_BUILTINS_BY_LANGUAGE, \
    is_builtin_entity
from snips_nlu.constants import UTTERANCES, DATA, ENTITY, ENTITIES, INTENTS
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


def default_features(language, dataset, intent, crf_features_config,
                     use_stemming, random_state,
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
                    "tagging_scheme_code": TaggingScheme.BIO.value},
                "offsets": [-2, -1, 0]
            }
        )

    # Entity lookup
    def preprocess(string):
        normalized = normalize(string)
        return stem(normalized, language) if use_stemming else normalized

    intent_entities = get_intent_custom_entities(dataset, intent)
    for entity_name, entity in intent_entities.iteritems():
        if not entity[UTTERANCES]:
            continue

        collection = list(
            set(preprocess(e) for e in entity[UTTERANCES].keys()))

        features.append(
            {
                "factory_name": "get_token_is_in_fn",
                "args": {
                    "tokens_collection": collection,
                    "collection_name": entity_name,
                    "use_stemming": use_stemming,
                    "tagging_scheme_code": TaggingScheme.BILOU.value,
                    "language_code": language.iso_code,
                },
                "offsets": tuple(crf_features_config.entities_offsets)
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
