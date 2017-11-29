from __future__ import unicode_literals

from snips_nlu.languages import Language
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features


def language_specific_features(dataset, intent, config):
    language = Language.KO
    features = default_features(language, dataset, intent, config,
                                use_stemming=False)

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
