from __future__ import unicode_literals

from snips_nlu.languages import Language
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features, default_shape_ngram_features


def language_specific_features(dataset, intent, config):
    language = Language.DE
    features = default_features(language, dataset, intent, config,
                                use_stemming=True)

    features += default_shape_ngram_features(language)
    return features
