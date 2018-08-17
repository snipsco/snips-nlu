from __future__ import print_function, unicode_literals

import json
from pathlib import Path

import plac

from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.utils import json_string


@plac.annotations(
    dataset_path=("Path to the dataset file", "positional", None, str),
    output_path=("Destination path for the json metrics", "positional", None,
                 str),
    nb_folds=("Number of folds to use for the cross-validation", "option", "n",
              int),
    train_size_ratio=("Fraction of the data that we want to use for training "
                      "(between 0 and 1)", "option", "t", float),
    exclude_slot_metrics=("Exclude slot metrics and slot errors in the output",
                          "flag", "s", bool),
    include_errors=("Include parsing errors in the output", "flag", "i", bool))
def cross_val_metrics(dataset_path, output_path, nb_folds=5,
                      train_size_ratio=1.0, exclude_slot_metrics=False,
                      include_errors=False):
    def progression_handler(progress):
        print("%d%%" % int(progress * 100))

    metrics_args = dict(
        dataset=dataset_path,
        engine_class=SnipsNLUEngine,
        progression_handler=progression_handler,
        nb_folds=nb_folds,
        train_size_ratio=train_size_ratio,
        include_slot_metrics=not exclude_slot_metrics,
    )

    with Path(dataset_path).open("r", encoding="utf8") as f:
        load_resources(json.load(f)["language"])

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
    exclude_slot_metrics=("Exclude slot metrics and slot errors in the output",
                          "flag", "s", bool),
    include_errors=("Include parsing errors in the output", "flag", "i", bool))
def train_test_metrics(train_dataset_path, test_dataset_path, output_path,
                       exclude_slot_metrics=False, include_errors=False):
    metrics_args = dict(
        train_dataset=train_dataset_path,
        test_dataset=test_dataset_path,
        engine_class=SnipsNLUEngine,
        include_slot_metrics=not exclude_slot_metrics
    )

    with Path(train_dataset_path).open("r", encoding="utf8") as f:
        load_resources(json.load(f)["language"])

    from snips_nlu_metrics import compute_train_test_metrics

    metrics = compute_train_test_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("parsing_errors")

    with Path(output_path).open(mode="w", encoding="utf8") as f:
        f.write(json_string(metrics))
