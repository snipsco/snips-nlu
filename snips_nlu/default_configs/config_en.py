CONFIG = {
    "unit_name": "nlu_engine",
    "intent_parsers_configs": [
        {
            "unit_name": "deterministic_intent_parser",
            "max_queries": 100,
            "max_pattern_length": 1000
        },
        {
            "unit_name": "probabilistic_intent_parser",
            "slot_filler_config": {
                "unit_name": "crf_slot_filler",
                "feature_factory_configs": [
                    {
                        "args": {
                            "common_words_gazetteer_name":
                                "top_10000_words_stemmed",
                            "use_stemming": True,
                            "n": 1
                        },
                        "factory_name": "ngram",
                        "offsets": [
                            -2,
                            -1,
                            0,
                            1,
                            2
                        ]
                    },
                    {
                        "args": {
                            "common_words_gazetteer_name":
                                "top_10000_words_stemmed",
                            "use_stemming": True,
                            "n": 2
                        },
                        "factory_name": "ngram",
                        "offsets": [
                            -2,
                            1
                        ]
                    },
                    {
                        "args": {},
                        "factory_name": "is_digit",
                        "offsets": [
                            -1,
                            0,
                            1
                        ]
                    },
                    {
                        "args": {},
                        "factory_name": "is_first",
                        "offsets": [
                            -2,
                            -1,
                            0
                        ]
                    },
                    {
                        "args": {},
                        "factory_name": "is_last",
                        "offsets": [
                            0,
                            1,
                            2
                        ]
                    },
                    {
                        "args": {
                            "n": 1
                        },
                        "factory_name": "shape_ngram",
                        "offsets": [
                            0
                        ]
                    },
                    {
                        "args": {
                            "n": 2
                        },
                        "factory_name": "shape_ngram",
                        "offsets": [
                            -1,
                            0
                        ]
                    },
                    {
                        "args": {
                            "n": 3
                        },
                        "factory_name": "shape_ngram",
                        "offsets": [
                            -1
                        ]
                    },
                    {
                        "args": {
                            "use_stemming": True,
                            "tagging_scheme_code": 2
                        },
                        "factory_name": "entity_match",
                        "offsets": [
                            -2,
                            -1,
                            0
                        ],
                        "drop_out": 0.5
                    },
                    {
                        "args": {
                            "tagging_scheme_code": 1
                        },
                        "factory_name": "builtin_entity_match",
                        "offsets": [
                            -2,
                            -1,
                            0
                        ]
                    },
                    {
                        "args": {
                            "cluster_name": "brown_clusters",
                            "use_stemming": False
                        },
                        "factory_name": "word_cluster",
                        "offsets": [
                            -2,
                            -1,
                            0,
                            1
                        ]
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
                    "add_builtin_entities_examples": True,
                    "unknown_word_prob": 0,
                    "unknown_words_replacement_string": None
                },
                "featurizer_config": {
                    "sublinear_tf": False,
                    "pvalue_threshold": 0.4,
                    "word_clusters_name": None,
                    "use_stemming": True
                },
                "random_seed": None
            }
        }
    ]
}
