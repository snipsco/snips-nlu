# coding=utf-8
from __future__ import unicode_literals, print_function

import argparse
import json
from builtins import input, bytes
from pathlib import Path

from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.pipeline.configs.nlu_engine import NLUEngineConfig


def debug_training(dataset_path, config_path=None):
    with Path(dataset_path).open("r", encoding="utf8") as f:
        dataset = json.load(f)

    config = None
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf8") as f:
            config = NLUEngineConfig.from_dict(json.load(f))

    engine = SnipsNLUEngine(config).fit(dataset)

    while True:
        query = input("Enter a query (type 'q' to quit): ").strip()
        if isinstance(query, bytes):
            query = query.decode("utf8")
        if query == "q":
            break
        print(json.dumps(engine.parse(query), indent=2))


def debug_inference(engine_path):
    engine = SnipsNLUEngine.from_path(engine_path)

    while True:
        query = input("Enter a query (type 'q' to quit): ").strip()
        if isinstance(query, bytes):
            query = query.decode("utf8")
        if query == "q":
            break
        print(json.dumps(engine.parse(query), indent=2))


def main_debug():
    parser = argparse.ArgumentParser(description="Debug snippets")
    parser.add_argument("mode", choices=["training", "inference"],
                        help="'training' to debug training and 'inference to "
                             "debug inference'")
    parser.add_argument("path",
                        help="Path to the dataset or trained assistant")
    parser.add_argument("--config-path",
                        help="Path to the assistant configuration")
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    if mode == "training":
        debug_training(*list(args.values()))
    elif mode == "inference":
        args.pop("config_path")
        debug_inference(*list(args.values()))
    else:
        raise ValueError("Invalid mode %s" % mode)


if __name__ == '__main__':
    main_debug()
