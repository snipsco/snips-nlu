from __future__ import print_function, unicode_literals


def add_train_parser(subparsers):
    subparser = subparsers.add_parser(
        "train", help="Train an NLU engine on a provided dataset")
    subparser.add_argument("dataset_path", type=str,
                           help="Path to the training dataset file")
    subparser.add_argument("output_path", type=str,
                           help="Path of the output model")
    subparser.add_argument("-c", "--config_path", type=str,
                           help="Path to the NLU engine configuration")
    subparser.add_argument("-r", "--random_seed", type=int,
                           help="Random seed to use for training")
    subparser.add_argument("-v", "--verbosity", action="count", default=0,
                           help="Increase output verbosity")
    subparser.set_defaults(func=_train)
    return subparser


def _train(args_namespace):
    return train(
        args_namespace.dataset_path, args_namespace.output_path,
        args_namespace.config_path, args_namespace.verbosity,
        args_namespace.random_seed)


def train(dataset_path, output_path, config_path=None, verbose=False,
          random_seed=None):
    """Train an NLU engine on the provided dataset"""
    import json
    import logging
    from pathlib import Path

    from snips_nlu import SnipsNLUEngine
    from snips_nlu.cli.utils import set_nlu_logger
    from snips_nlu.common.utils import check_random_state

    if verbose == 1:
        set_nlu_logger(logging.INFO)
    elif verbose >= 2:
        set_nlu_logger(logging.DEBUG)

    with Path(dataset_path).open("r", encoding="utf8") as f:
        dataset = json.load(f)

    config = None
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf8") as f:
            config = json.load(f)

    random_state = check_random_state(random_seed)

    print("Create and train the engine...")
    engine = SnipsNLUEngine(config, random_state=random_state).fit(dataset)

    print("Persisting the engine...")
    engine.persist(output_path)

    print("Saved the trained engine to %s" % output_path)
