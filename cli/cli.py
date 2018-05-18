from __future__ import print_function, unicode_literals

import argparse
import io
import json
import os
import sys

from builtins import bytes, input
from pygments import highlight
from pygments.formatters.terminal import TerminalFormatter
from pygments.lexers.data import JsonLexer

from snips_nlu import (SnipsNLUEngine, load_resources)

JSON_LEXER = JsonLexer()
TERMINAL_FORMATTER = TerminalFormatter(bg="dark")


def parse_train_args(args):
    parser = argparse.ArgumentParser("Train an NLU engine and persist it in "
                                     "a json file")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-c", "--config-path", type=str, metavar="",
                        help="Path to the NLU engine configuration")
    return parser.parse_args(args)


def main_train_engine():
    args = vars(parse_train_args(sys.argv[1:]))

    dataset_path = args.pop("dataset_path")
    with io.open(dataset_path, "r", encoding="utf8") as f:
        dataset = json.load(f)

    config = None
    if args.get("config_path") is not None:
        config_path = args.pop("config_path")
        with io.open(config_path, "r", encoding="utf8") as f:
            config = json.load(f)

    load_resources(dataset["language"])
    engine = SnipsNLUEngine(config).fit(dataset)
    print("Create and train the engine...")

    output_path = args.pop("output_path")
    serialized_engine = bytes(json.dumps(engine.to_dict()), encoding="utf8")
    with io.open(output_path, "w", encoding="utf8") as f:
        f.write(serialized_engine.decode("utf8"))
    print("Saved the trained engine to %s" % output_path)


def parse_inference_args(args):
    parser = argparse.ArgumentParser("Load a trained NLU engine and play with "
                                     "its parsing API")
    parser.add_argument("training_path", type=str,
                        help="Path to a json-serialized trained engine")
    return parser.parse_args(args)


def main_engine_inference():
    args = vars(parse_inference_args(sys.argv[1:]))

    training_path = args.pop("training_path")
    with io.open(os.path.abspath(training_path), "r", encoding="utf8") as f:
        engine_dict = json.load(f)
    language = engine_dict["dataset_metadata"]["language_code"]
    load_resources(language)
    engine = SnipsNLUEngine.from_dict(engine_dict)

    while True:
        query = input("Enter a query (type 'q' to quit): ").strip()
        if isinstance(query, bytes):
            query = query.decode("utf8")
        if query == "q":
            break
        json_dump = json.dumps(engine.parse(query), sort_keys=True, indent=2)
        colorful_json = highlight(json_dump, JSON_LEXER, TERMINAL_FORMATTER)
        print(colorful_json)


def parse_cross_val_args(args):
    parser = argparse.ArgumentParser("Compute cross validation metrics on a "
                                     "given dataset")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-n", "--nb-folds", type=int, metavar="",
                        help="Number of folds to use for the cross-validation")
    parser.add_argument("-t", "--train-size-ratio", type=float, metavar="",
                        help="Fraction of the data that we want to use for "
                             "training (between 0 and 1)")
    parser.add_argument(
        "-esm", "--exclude-slot-metrics", action="store_true",
        help="Exclude slot metrics and slot errors in the output")
    parser.add_argument("-i", "--include-errors", action="store_true",
                        help="Include parsing errors in the output")
    return parser.parse_args(args)


def main_cross_val_metrics():
    args = vars(parse_cross_val_args(sys.argv[1:]))

    dataset_path = args.pop("dataset_path")
    output_path = args.pop("output_path")
    exclude_slot_metrics = args.get("exclude_slot_metrics", False)

    def progression_handler(progress):
        print("%d%%" % int(progress * 100))

    metrics_args = dict(
        dataset=dataset_path,
        engine_class=SnipsNLUEngine,
        include_slot_metrics=not exclude_slot_metrics,
        progression_handler=progression_handler
    )
    if args.get("nb_folds") is not None:
        nb_folds = args.pop("nb_folds")
        metrics_args.update(dict(nb_folds=nb_folds))
    if args.get("train_size_ratio") is not None:
        train_size_ratio = args.pop("train_size_ratio")
        metrics_args.update(dict(train_size_ratio=train_size_ratio))

    include_errors = args.get("include_errors", False)

    with io.open(dataset_path, "r", encoding="utf-8") as f:
        load_resources(json.load(f)["language"])

    from snips_nlu_metrics import compute_cross_val_metrics

    metrics = compute_cross_val_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("parsing_errors")

    with io.open(output_path, mode="w") as f:
        json_dump = json.dumps(metrics, sort_keys=True, indent=2)
        f.write(bytes(json_dump, encoding="utf8").decode("utf8"))


def parse_train_test_args(args):
    parser = argparse.ArgumentParser("Compute train/test metrics on a given "
                                     "pair training set/testing set")
    parser.add_argument("train_dataset_path", type=str)
    parser.add_argument("test_dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument(
        "-s", "--exclude-slot-metrics", action="store_true",
        help="Exclude slot metrics and slot errors in the output")
    parser.add_argument("-i", "--include-errors", action="store_true",
                        help="Include parsing errors in the output")
    return parser.parse_args(args)


def main_train_test_metrics():
    args = vars(parse_train_test_args(sys.argv[1:]))

    train_dataset_path = args.pop("train_dataset_path")
    test_dataset_path = args.pop("test_dataset_path")
    output_path = args.pop("output_path")
    exclude_slot_metrics = args.get("exclude_slot_metrics", False)

    metrics_args = dict(
        train_dataset=train_dataset_path,
        test_dataset=test_dataset_path,
        engine_class=SnipsNLUEngine,
        include_slot_metrics=not exclude_slot_metrics
    )

    include_errors = args.get("include_errors", False)
    with io.open(train_dataset_path, "r", encoding="utf-8") as f:
        load_resources(json.load(f)["language"])

    from snips_nlu_metrics import compute_train_test_metrics

    metrics = compute_train_test_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("parsing_errors")

    with io.open(output_path, mode="w") as f:
        json_dump = json.dumps(metrics, sort_keys=True, indent=2)
        f.write(bytes(json_dump, encoding="utf8").decode("utf8"))
