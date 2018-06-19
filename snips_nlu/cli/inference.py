from __future__ import unicode_literals

import json
from pathlib import Path

import plac

from snips_nlu import load_resources, SnipsNLUEngine


@plac.annotations(
    training_path=("Path to a trained engine", "positional", None, str),
    query=("Query to parse. If provided, it disables the interactive "
           "behavior.", "option", "q", str))
def parse(training_path, query):
    """Load a trained NLU engine and play with its parsing API interactively"""
    training_path = Path(training_path)
    with training_path.open("r", encoding="utf8") as f:
        engine_dict = json.load(f)
    language = engine_dict["dataset_metadata"]["language_code"]
    load_resources(language)
    engine = SnipsNLUEngine.from_dict(engine_dict)

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
