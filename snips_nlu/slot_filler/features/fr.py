features = [
    {
        "args": {
            "common_words_gazetteer_name": "top_10000_words",
            "language_code": "fr",
            "use_stemming": True,
            "n": 1
        },
        "factory_name": "get_ngram_fn",
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
            "common_words_gazetteer_name": "top_10000_words",
            "language_code": "fr",
            "use_stemming": True,
            "n": 2
        },
        "factory_name": "get_ngram_fn",
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
            "tagging_scheme_code": 1,
            "built_in_entity_label": "snips/ordinal",
            "language_code": "fr"
        },
        "factory_name": "get_built_in_annotation_fn",
        "offsets": [
            -2,
            -1,
            0
        ]
    },
    {
        "args": {
            "tagging_scheme_code": 1,
            "built_in_entity_label": "snips/temperature",
            "language_code": "fr"
        },
        "factory_name": "get_built_in_annotation_fn",
        "offsets": [
            -2,
            -1,
            0
        ]
    },
    {
        "args": {
            "tagging_scheme_code": 1,
            "built_in_entity_label": "snips/amountOfMoney",
            "language_code": "fr"
        },
        "factory_name": "get_built_in_annotation_fn",
        "offsets": [
            -2,
            -1,
            0
        ]
    },
    {
        "args": {
            "tagging_scheme_code": 1,
            "built_in_entity_label": "snips/number",
            "language_code": "fr"
        },
        "factory_name": "get_built_in_annotation_fn",
        "offsets": [
            -2,
            -1,
            0
        ]
    },
    {
        "args": {
            "tagging_scheme_code": 1,
            "built_in_entity_label": "snips/datetime",
            "language_code": "fr"
        },
        "factory_name": "get_built_in_annotation_fn",
        "offsets": [
            -2,
            -1,
            0
        ]
    },
    {
        "args": {
            "tagging_scheme_code": 1,
            "built_in_entity_label": "snips/duration",
            "language_code": "fr"
        },
        "factory_name": "get_built_in_annotation_fn",
        "offsets": [
            -2,
            -1,
            0
        ]
    },
    {
        "args": {
            "tagging_scheme_code": 1,
            "built_in_entity_label": "snips/percentage",
            "language_code": "fr"
        },
        "factory_name": "get_built_in_annotation_fn",
        "offsets": [
            -2,
            -1,
            0
        ]
    },
    {
        "args": {
            "language_code": "fr",
            "n": 1
        },
        "factory_name": "get_shape_ngram_fn",
        "offsets": [
            0
        ]
    },
    {
        "args": {
            "language_code": "fr",
            "n": 2
        },
        "factory_name": "get_shape_ngram_fn",
        "offsets": [
            -1,
            0
        ]
    },
    {
        "args": {
            "language_code": "fr",
            "n": 3
        },
        "factory_name": "get_shape_ngram_fn",
        "offsets": [
            -1
        ]
    },
    # {
    #     "args": {
    #         "use_stemming": True,
    #         "collection_name": "Temperature",
    #         "tokens_collection": [
    #             "boiling",
    #             "cold",
    #             "hot",
    #             "iced"
    #         ],
    #         "tagging_scheme_code": 2,
    #         "language_code": "fr"
    #     },
    #     "factory_name": "get_token_is_in_fn",
    #     "offsets": [
    #         -2,
    #         -1,
    #         0
    #     ]
    # },
]
