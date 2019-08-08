from __future__ import print_function


def add_version_parser(subparsers, formatter_class):
    from snips_nlu.__about__ import __version__
    subparser = subparsers.add_parser(
        "version", formatter_class=formatter_class,
        help="Print the package version")
    subparser.set_defaults(func=lambda _: print(__version__))
    return subparser


def add_model_version_parser(subparsers, formatter_class):
    from snips_nlu.__about__ import __model_version__
    subparser = subparsers.add_parser(
        "model-version", formatter_class=formatter_class,
        help="Print the model version")
    subparser.set_defaults(func=lambda _: print(__model_version__))
    return subparser
