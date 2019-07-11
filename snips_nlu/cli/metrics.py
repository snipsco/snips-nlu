from __future__ import print_function, unicode_literals


def make_engine_cls(config):
    from snips_nlu import SnipsNLUEngine
    from snips_nlu_metrics import Engine

    class ConfigEngine(Engine):
        def __init__(self):
            self.engine = None
            self.config = config

        def fit(self, dataset):
            self.engine = SnipsNLUEngine(self.config).fit(dataset)
            return self

        def parse(self, text):
            return self.engine.parse(text)

    return ConfigEngine


def add_cross_val_metrics_parser(subparsers):
    subparser = subparsers.add_parser(
        "cross-val-metrics",
        help="Compute cross-validation metrics on a given dataset")
    subparser.add_argument("dataset_path", type=str,
                           help="Path to the dataset file")
    subparser.add_argument("output_path", type=str,
                           help="Destination path for the json metrics")
    subparser.add_argument("-c", "--config_path", type=str,
                           help="Path to a NLU engine config file")
    subparser.add_argument("-n", "--nb_folds", type=int, default=5,
                           help="Number of folds to use for the "
                                "cross-validation")
    subparser.add_argument("-t", "--train_size_ratio", default=1.0, type=float,
                           help="Fraction of the data that we want to use for "
                                "training (between 0 and 1)")
    subparser.add_argument("-s", "--exclude_slot_metrics", action="store_true",
                           help="Fraction of the data that we want to use for "
                                "training (between 0 and 1)")
    subparser.add_argument("-i", "--include_errors", action="store_true",
                           help="Include parsing errors in the output")
    subparser.add_argument("-v", "--verbosity", action="count", default=0,
                           help="Increase output verbosity")
    subparser.set_defaults(func=_cross_val_metrics)
    return subparser


def _cross_val_metrics(args_namespace):
    return cross_val_metrics(
        args_namespace.dataset_path, args_namespace.output_path,
        args_namespace.config_path, args_namespace.nb_folds,
        args_namespace.train_size_ratio, args_namespace.exclude_slot_metrics,
        args_namespace.include_errors, args_namespace.verbosity)


def cross_val_metrics(dataset_path, output_path, config_path=None, nb_folds=5,
                      train_size_ratio=1.0, exclude_slot_metrics=False,
                      include_errors=False, verbose=0):
    import json
    import logging
    from pathlib import Path
    from snips_nlu_metrics import compute_cross_val_metrics
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.cli.utils import set_nlu_logger
    from snips_nlu.common.utils import json_string

    if verbose == 1:
        set_nlu_logger(logging.INFO)
    elif verbose >= 2:
        set_nlu_logger(logging.DEBUG)

    def progression_handler(progress):
        print("%d%%" % int(progress * 100))

    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as f:
            config = json.load(f)
        engine_cls = make_engine_cls(config)
    else:
        engine_cls = SnipsNLUEngine

    metrics_args = dict(
        dataset=dataset_path,
        engine_class=engine_cls,
        progression_handler=progression_handler,
        nb_folds=nb_folds,
        train_size_ratio=train_size_ratio,
        include_slot_metrics=not exclude_slot_metrics,
        slot_matching_lambda=_match_trimmed_values
    )

    metrics = compute_cross_val_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("parsing_errors")

    with Path(output_path).open(mode="w", encoding="utf8") as f:
        f.write(json_string(metrics))


def add_train_test_metrics_parser(subparsers):
    subparser = subparsers.add_parser(
        "train-test-metrics",
        help="Compute NLU metrics training on a given dataset and testing on "
             "another")
    subparser.add_argument("train_dataset_path", type=str,
                           help="Path to the dataset used for training")
    subparser.add_argument("test_dataset_path", type=str,
                           help="Path to the dataset used for testing")
    subparser.add_argument("output_path", type=str,
                           help="Destination path for the json metrics")
    subparser.add_argument("-c", "--config_path", type=str,
                           help="Path to a NLU engine config file")
    subparser.add_argument("-s", "--exclude_slot_metrics", action="store_true",
                           help="Fraction of the data that we want to use for "
                                "training (between 0 and 1)")
    subparser.add_argument("-i", "--include_errors", action="store_true",
                           help="Include parsing errors in the output")
    subparser.add_argument("-v", "--verbosity", action="count", default=0,
                           help="Increase output verbosity")
    subparser.set_defaults(func=_train_test_metrics)
    return subparser


def _train_test_metrics(args_namespace):
    return train_test_metrics(
        args_namespace.train_dataset_path, args_namespace.test_dataset_path,
        args_namespace.output_path, args_namespace.config_path,
        args_namespace.exclude_slot_metrics, args_namespace.include_errors,
        args_namespace.verbosity)


def train_test_metrics(train_dataset_path, test_dataset_path, output_path,
                       config_path=None, exclude_slot_metrics=False,
                       include_errors=False, verbosity=0):
    import json
    import logging
    from pathlib import Path
    from snips_nlu_metrics import compute_train_test_metrics
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.cli.utils import set_nlu_logger
    from snips_nlu.common.utils import json_string

    if verbosity == 1:
        set_nlu_logger(logging.INFO)
    elif verbosity >= 2:
        set_nlu_logger(logging.DEBUG)

    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as f:
            config = json.load(f)
        engine_cls = make_engine_cls(config)
    else:
        engine_cls = SnipsNLUEngine

    metrics_args = dict(
        train_dataset=train_dataset_path,
        test_dataset=test_dataset_path,
        engine_class=engine_cls,
        include_slot_metrics=not exclude_slot_metrics,
        slot_matching_lambda=_match_trimmed_values
    )

    metrics = compute_train_test_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("parsing_errors")

    with Path(output_path).open(mode="w", encoding="utf8") as f:
        f.write(json_string(metrics))


def _match_trimmed_values(lhs_slot, rhs_slot):
    lhs_value = lhs_slot["text"].strip()
    rhs_value = rhs_slot["rawValue"].strip()
    return lhs_value == rhs_value
