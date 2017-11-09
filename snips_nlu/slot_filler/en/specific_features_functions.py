from __future__ import unicode_literals

from snips_nlu.languages import Language
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features, default_shape_ngram_features


def language_specific_features(dataset, intent, crf_features_config):
    language = Language.EN
    features = default_features(language, dataset, intent, crf_features_config,
                                use_stemming=True)

    features += default_shape_ngram_features(language)

    features.append({
        "factory_name": "get_word_cluster_fn",
        "args": {"cluster_name": "brown_clusters",
                 "language_code": language.iso_code,
                 "use_stemming": False},
        "offsets": (-2, -1, 0, 1)
    })

    return features
