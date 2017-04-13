from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_resources import get_gazetteer
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features


def language_specific_features(module_name, intent_entities):
    """
    :param module_name: name of the module in which feature functions are 
    defined 
    :param intent_entities: dict containing entities for the related intent
    """
    language = Language.EN
    common_words = get_gazetteer(language, "top_10000_words")
    features = default_features(module_name, language, intent_entities,
                                use_stemming=True,
                                entities_offsets=(-2, -1, 0),
                                entity_keep_prob=.5, common_words=common_words)

    gazetteer_names = ["pois", "cities_us", "cities_world", "countries",
                       "regions", "states_us", "stop_words",
                       "street_identifier"]

    for gazetteer_name in gazetteer_names:
        features.append({
            "module_name": module_name,
            "factory_name": "get_is_in_gazetteer_fn",
            "args": {"gazetteer_name": gazetteer_name,
                     "language_code": language.iso_code},
            "offsets": (-1, 0, 1)
        })

    features.append({
        "module_name": module_name,
        "factory_name": "get_word_cluster_fn",
        "args": {"cluster_name": "brown_clusters",
                 "language_code": language.iso_code},
        "offsets": (-2, -1, 0, 1)
    })

    return features
