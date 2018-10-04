def default_features_factories():
    """These are the default features used by the :class:`.CRFSlotFiller`
        objects"""

    from snips_nlu.slot_filler.crf_utils import TaggingScheme
    from snips_nlu.slot_filler.feature_factory import (
        NgramFactory, IsDigitFactory, IsFirstFactory, IsLastFactory,
        ShapeNgramFactory, CustomEntityMatchFactory, BuiltinEntityMatchFactory)

    return [
        {
            "args": {
                "common_words_gazetteer_name": None,
                "use_stemming": False,
                "n": 1
            },
            "factory_name": NgramFactory.name,
            "offsets": [-2, -1, 0, 1, 2]
        },
        {
            "args": {
                "common_words_gazetteer_name": None,
                "use_stemming": False,
                "n": 2
            },
            "factory_name": NgramFactory.name,
            "offsets": [-2, 1]
        },
        {
            "args": {},
            "factory_name": IsDigitFactory.name,
            "offsets": [-1, 0, 1]
        },
        {
            "args": {},
            "factory_name": IsFirstFactory.name,
            "offsets": [-2, -1, 0]
        },
        {
            "args": {},
            "factory_name": IsLastFactory.name,
            "offsets": [0, 1, 2]
        },
        {
            "args": {
                "n": 1
            },
            "factory_name": ShapeNgramFactory.name,
            "offsets": [0]
        },
        {
            "args": {
                "n": 2
            },
            "factory_name": ShapeNgramFactory.name,
            "offsets": [-1, 0]
        },
        {
            "args": {
                "n": 3
            },
            "factory_name": ShapeNgramFactory.name,
            "offsets": [-1]
        },
        {
            "args": {
                "use_stemming": False,
                "tagging_scheme_code": TaggingScheme.BILOU.value,
            },
            "factory_name": CustomEntityMatchFactory.name,
            "offsets": [-2, -1, 0],
            "drop_out": 0.5
        },
        {
            "args": {
                "tagging_scheme_code": TaggingScheme.BIO.value,
            },
            "factory_name": BuiltinEntityMatchFactory.name,
            "offsets": [-2, -1, 0]
        },
    ]
