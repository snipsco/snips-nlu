def language_specific_features(module_name):
    """
    :param module_name: name of the module in which feature functions are 
    defined 
    """
    features = [
        {
            "module_name": module_name,
            "factory_name": "get_prefix_fn",
            "args": {"prefix_size": 1},
            "offsets": [0]
        },
        {
            "module_name": module_name,
            "factory_name": "get_prefix_fn",
            "args": {"prefix_size": 2},
            "offsets": [0]
        },
        {
            "module_name": module_name,
            "factory_name": "get_suffix_fn",
            "args": {"suffix_size": 1},
            "offsets": [0]
        },
        {
            "module_name": module_name,
            "factory_name": "get_suffix_fn",
            "args": {"suffix_size": 2},
            "offsets": [0]
        },
    ]

    return features
