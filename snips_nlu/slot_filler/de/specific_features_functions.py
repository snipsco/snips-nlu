from __future__ import unicode_literals

from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features, default_shape_ngram_features


def language_specific_features(crf_features_config, intent_entities):
    """
    :param intent_entities: dict containing entities for the related intent
    """
    language = Language.DE
    features = default_features(language, intent_entities,
                                crf_features_config, use_stemming=True)

    gazetteer_names = ["cities_germany", "cities_world", "countries",
                       "lander_germany", "street_identifier"]
    features += default_shape_ngram_features(language)
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
