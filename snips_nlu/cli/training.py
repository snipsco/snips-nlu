from __future__ import print_function, unicode_literals

import json
import logging
from pathlib import Path

import plac

from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.cli.utils import set_nlu_logger


@plac.annotations(
    dataset_path=("Path to the training dataset file", "positional", None,
                  str),
    output_path=("Path of the output model", "positional", None, str),
    config_path=("Path to the NLU engine configuration", "option", "c", str),
    verbose=("Print logs", "flag", "v"),
)
def train(dataset_path, output_path, config_path, verbose):
    """Train an NLU engine on the provided dataset"""
    if verbose:
        set_nlu_logger(logging.INFO)
    with Path(dataset_path).open("r", encoding="utf8") as f:
        dataset = json.load(f)

    config = None
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf8") as f:
            config = json.load(f)

    load_resources(dataset["language"])
    print("Create and train the engine...")
    engine = SnipsNLUEngine(config).fit(dataset)

    print("Persisting the engine...")
    engine.persist(output_path)

    print("Saved the trained engine to %s" % output_path)
