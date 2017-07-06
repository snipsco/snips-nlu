from __future__ import unicode_literals

from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features


def language_specific_features(intent_entities):
    """
    :param intent_entities: dict containing entities for the related intent
    """
    language = Language.DE
    features = default_features(language, intent_entities, use_stemming=True,
                                entities_offsets=(-2, -1, 0),
                                entity_keep_prob=.5,
                                common_words_gazetteer_name="top_10000_words")

    gazetteer_names = ["cities_germany", "cities_world", "countries",
                       "lander_germany", "street_identifier"]

    for gazetteer_name in gazetteer_names:
        features.append({
            "factory_name": "get_is_in_gazetteer_fn",
            "args": {"gazetteer_name": gazetteer_name,
                     "language_code": language.iso_code,
                     "tagging_scheme_code": TaggingScheme.BILOU.value,
                     "use_stemming": False},
            "offsets": (-1, 0, 1)
        })

    features.append({
        "factory_name": "get_word_cluster_fn",
        "args": {"cluster_name": "brown_clusters",
                 "language_code": language.iso_code,
                 "use_stemming": False},
        "offsets": (-2, -1, 0, 1)
    })

    return features
