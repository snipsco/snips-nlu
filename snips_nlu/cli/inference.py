from __future__ import unicode_literals, print_function

import logging
from builtins import input

import plac

from snips_nlu import SnipsNLUEngine
from snips_nlu.cli.utils import set_nlu_logger
from snips_nlu.common.utils import unicode_string, json_string


@plac.annotations(
    training_path=("Path to a trained engine", "positional", None, str),
    query=("Query to parse. If provided, it disables the interactive "
           "behavior.", "option", "q", str),
    verbose=("Print logs", "flag", "v"),
)
def parse(training_path, query, verbose=False):
    """Load a trained NLU engine and play with its parsing API interactively"""
    from builtins import str
    if verbose:
        set_nlu_logger(logging.DEBUG)

    engine = SnipsNLUEngine.from_path(training_path)

    if query:
        print_parsing_result(engine, query)
        return

    while True:
        query = input("Enter a query (type 'q' to quit): ").strip()
        if not isinstance(query, str):
            query = query.decode("utf-8")
        if query == "q":
            break
        print_parsing_result(engine, query)


def print_parsing_result(engine, query):
    query = unicode_string(query)
    json_dump = json_string(engine.parse(query), sort_keys=True, indent=2)
    print(json_dump)
