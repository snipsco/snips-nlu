from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features


def language_specific_features(module_name, intent_entities):
    """
    :param module_name: name of the module in which feature functions are 
    defined 
    :param intent_entities: dict containing entities for the related intent
    """
    language = Language.EN
    features = default_features(module_name, language, intent_entities,
                                use_stemming=True,
                                entities_offsets=(-2, -1, 0),
                                entity_keep_prob=.5,
                                common_words_gazetteer_name="top_10000_words")

    gazetteer_names = ["cities_us", "cities_world", "countries",
                       "states_us", "street_identifier"]

    for gazetteer_name in gazetteer_names:
        features.append({
            "module_name": module_name,
            "factory_name": "get_is_in_gazetteer_fn",
            "args": {"gazetteer_name": gazetteer_name,
                     "language_code": language.iso_code,
                     "tagging_scheme_code": TaggingScheme.BILOU.value,
                     "use_stemming": False},
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
