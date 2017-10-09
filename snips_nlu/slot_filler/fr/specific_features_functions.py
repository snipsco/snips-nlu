from __future__ import unicode_literals

from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features, default_shape_ngram_features


def language_specific_features(dataset, intent, config):
    """
    :param intent_entities: dict containing entities for the related intent
    """
    language = Language.FR
    features = default_features(language, dataset, intent, config,
                                use_stemming=True,
                                common_words_gazetteer_name="top_10000_words")
    features += default_shape_ngram_features(language)
    gazetteer_names = ["cities_france", "cities_world", "countries",
                       "departements_france", "regions_france",
                       "street_identifier"]

    for gazetteer_name in gazetteer_names:
        features.append({
            "factory_name": "get_is_in_gazetteer_fn",
            "args": {
                "gazetteer_name": gazetteer_name,
                "language_code": language.iso_code,
                "tagging_scheme_code": TaggingScheme.BILOU.value,
                "use_stemming": False
            },
            "offsets": (-1, 0, 1)
        })

    return features
