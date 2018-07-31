from __future__ import unicode_literals, print_function

import json
from builtins import input

import plac

from snips_nlu import SnipsNLUEngine


@plac.annotations(
    training_path=("Path to a trained engine", "positional", None, str),
    query=("Query to parse. If provided, it disables the interactive "
           "behavior.", "option", "q", str))
def parse(training_path, query):
    """Load a trained NLU engine and play with its parsing API interactively"""
    engine = SnipsNLUEngine.from_path(training_path)

    if query:
        print_parsing_result(engine, query)
        return

    while True:
        query = input("Enter a query (type 'q' to quit): ").strip()
        if query == "q":
            break
        print_parsing_result(engine, query)


def print_parsing_result(engine, query):
    if isinstance(query, bytes):
        query = query.decode("utf8")
    json_dump = json.dumps(engine.parse(query), sort_keys=True, indent=2)
    print(json_dump)
