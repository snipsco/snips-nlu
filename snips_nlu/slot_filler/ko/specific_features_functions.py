from snips_nlu.languages import Language
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features


def language_specific_features(intent_entities):
    """
    :param intent_entities: dict containing entities for the related intent
    """
    language = Language.KO
    features = default_features(language, intent_entities, use_stemming=False,
                                entities_offsets=(-2, -1, 0),
                                entity_keep_prob=.5)

    features += [
        {
            "factory_name": "get_prefix_fn",
            "args": {"prefix_size": 1},
            "offsets": [0]
        },
        {
            "factory_name": "get_prefix_fn",
            "args": {"prefix_size": 2},
            "offsets": [0]
        },
        {
            "factory_name": "get_suffix_fn",
            "args": {"suffix_size": 1},
            "offsets": [0]
        },
        {
            "factory_name": "get_suffix_fn",
            "args": {"suffix_size": 2},
            "offsets": [0]
        },
    ]

    return features
