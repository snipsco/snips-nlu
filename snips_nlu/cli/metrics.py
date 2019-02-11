from __future__ import print_function, unicode_literals

import json
import logging
from pathlib import Path

import plac

from snips_nlu import SnipsNLUEngine
from snips_nlu.cli.utils import set_nlu_logger
from snips_nlu.common.utils import json_string


def make_engine_cls(config):
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


@plac.annotations(
    dataset_path=("Path to the dataset file", "positional", None, str),
    output_path=("Destination path for the json metrics", "positional", None,
                 str),
    config_path=("Path to a NLU engine config file", "option", "c", str),
    nb_folds=("Number of folds to use for the cross-validation", "option", "n",
              int),
    train_size_ratio=("Fraction of the data that we want to use for training "
                      "(between 0 and 1)", "option", "t", float),
    exclude_slot_metrics=("Exclude slot metrics and slot errors in the output",
                          "flag", "s", bool),
    include_errors=("Include parsing errors in the output", "flag", "i", bool),
    verbose=("Print logs", "flag", "v"),
)
def cross_val_metrics(dataset_path, output_path, config_path=None, nb_folds=5,
                      train_size_ratio=1.0, exclude_slot_metrics=False,
                      include_errors=False, verbose=False):
    if verbose:
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

    from snips_nlu_metrics import compute_cross_val_metrics

    metrics = compute_cross_val_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("parsing_errors")

    with Path(output_path).open(mode="w", encoding="utf8") as f:
        f.write(json_string(metrics))


@plac.annotations(
    train_dataset_path=("Path to the dataset used for training", "positional",
                        None, str),
    test_dataset_path=("Path to the dataset used for testing", "positional",
                       None, str),
    output_path=("Destination path for the json metrics", "positional", None,
                 str),
    config_path=("Path to a NLU engine config file", "option", "c", str),
    exclude_slot_metrics=("Exclude slot metrics and slot errors in the output",
                          "flag", "s", bool),
    include_errors=("Include parsing errors in the output", "flag", "i", bool),
    verbose=("Print logs", "flag", "v"),
)
def train_test_metrics(train_dataset_path, test_dataset_path, output_path,
                       config_path=None, exclude_slot_metrics=False,
                       include_errors=False, verbose=False):
    if verbose:
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

    from snips_nlu_metrics import compute_train_test_metrics

    metrics = compute_train_test_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("parsing_errors")

    with Path(output_path).open(mode="w", encoding="utf8") as f:
        f.write(json_string(metrics))


def _match_trimmed_values(lhs_slot, rhs_slot):
    lhs_value = lhs_slot["text"].strip()
    rhs_value = rhs_slot["rawValue"].strip()
    return lhs_value == rhs_value
