from __future__ import unicode_literals, print_function


def add_parse_parser(subparsers):
    subparser = subparsers.add_parser(
        "parse", help="Load a trained NLU engine and perform parsing")
    subparser.add_argument("training_path", type=str,
                           help="Path to a trained engine")
    subparser.add_argument("-q", "--query", type=str,
                           help="Query to parse. If provided, it disables the "
                                "interactive behavior.")
    subparser.add_argument("-v", "--verbosity", action="count", default=0,
                           help="Increase output verbosity")
    subparser.set_defaults(func=_parse)
    return subparser


def _parse(args_namespace):
    return parse(args_namespace.training_path, args_namespace.query,
                 args_namespace.verbosity)


def parse(training_path, query, verbose=False):
    """Load a trained NLU engine and play with its parsing API interactively"""
    import logging
    from builtins import input, str
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.cli.utils import set_nlu_logger

    if verbose == 1:
        set_nlu_logger(logging.INFO)
    elif verbose >= 2:
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
    from snips_nlu.common.utils import unicode_string, json_string

    query = unicode_string(query)
    json_dump = json_string(engine.parse(query), sort_keys=True, indent=2)
    print(json_dump)
