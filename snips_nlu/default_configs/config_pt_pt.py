from __future__ import unicode_literals

CONFIG = {
    "unit_name": "nlu_engine",
    "intent_parsers_configs": [
        {
            "unit_name": "deterministic_intent_parser",
            "max_queries": 500,
            "max_pattern_length": 1000,
            "ignore_stop_words": True
        },
        {
            "unit_name": "probabilistic_intent_parser",
            "slot_filler_config": {
                "unit_name": "crf_slot_filler",
                "feature_factory_configs": [
                    {
                        "args": {
                            "common_words_gazetteer_name":
                                "top_5000_words_stemmed",
                            "use_stemming": True,
                            "n": 1
                        },
                        "factory_name": "ngram",
                        "offsets": [-2, -1, 0, 1, 2]
                    },
                    {
                        "args": {
                            "common_words_gazetteer_name":
                                "top_5000_words_stemmed",
                            "use_stemming": True,
                            "n": 2
                        },
                        "factory_name": "ngram",
                        "offsets": [-2, 1]
                    },
                    {
                        "args": {},
                        "factory_name": "is_digit",
                        "offsets": [-1, 0, 1]
                    },
                    {
                        "args": {},
                        "factory_name": "is_first",
                        "offsets": [-2, -1, 0]
                    },
                    {
                        "args": {},
                        "factory_name": "is_last",
                        "offsets": [0, 1, 2]
                    },
                    {
                        "args": {"n": 1},
                        "factory_name": "shape_ngram",
                        "offsets": [0]
                    },
                    {
                        "args": {"n": 2},
                        "factory_name": "shape_ngram",
                        "offsets": [-1, 0]
                    },
                    {
                        "args": {"n": 3},
                        "factory_name": "shape_ngram",
                        "offsets": [-1]
                    },
                    {
                        "args": {
                            "use_stemming": True,
                            "tagging_scheme_code": 2
                        },
                        "factory_name": "entity_match",
                        "offsets": [-2, -1, 0],
                        "drop_out": 0.5
                    },
                    {
                        "args": {"tagging_scheme_code": 1},
                        "factory_name": "builtin_entity_match",
                        "offsets": [-2, -1, 0]
                    }
                ],
                "crf_args": {
                    "c1": 0.1,
                    "c2": 0.1,
                    "algorithm": "lbfgs"
                },
                "tagging_scheme": 1,
                "data_augmentation_config": {
                    "min_utterances": 200,
                    "capitalization_ratio": 0.2,
                    "add_builtin_entities_examples": True
                },
                "random_seed": None
            },
            "intent_classifier_config": {
                "unit_name": "log_reg_intent_classifier",
                "data_augmentation_config": {
                    "min_utterances": 20,
                    "noise_factor": 5,
                    "add_builtin_entities_examples": False,
                    "max_unknown_words": None,
                    "unknown_word_prob": 0.0,
                    "unknown_words_replacement_string": None
                },
                "featurizer_config": {
                    "unit_name": "featurizer",
                    "pvalue_threshold": 0.4,
                    "added_cooccurrence_feature_ratio": 0.0,
                    "tfidf_vectorizer_config": {
                        "unit_name": "tfidf_vectorizer",
                        "use_stemming": True,
                        "word_clusters_name": None
                    },
                    "cooccurrence_vectorizer_config": {
                        "unit_name": "cooccurrence_vectorizer",
                        "window_size": None,
                        "filter_stop_words": True,
                        "unknown_words_replacement_string": None,
                        "keep_order": True
                    }
                },
                "random_seed": None
            }
        }
    ]
}
