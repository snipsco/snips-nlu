import argparse


class Formatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super(Formatter, self).__init__(prog, max_help_position=35, width=150)


def get_arg_parser():
    from snips_nlu.cli.download import (
        add_download_parser, add_download_all_languages_parser)
    from snips_nlu.cli.download_entity import (
        add_download_entity_parser, add_download_language_entities_parser)
    from snips_nlu.cli.generate_dataset import add_generate_dataset_subparser
    from snips_nlu.cli.inference import add_parse_parser
    from snips_nlu.cli.link import add_link_parser
    from snips_nlu.cli.metrics import (
        add_cross_val_metrics_parser, add_train_test_metrics_parser)
    from snips_nlu.cli.training import add_train_parser
    from snips_nlu.cli.versions import (
        add_version_parser, add_model_version_parser)

    arg_parser = argparse.ArgumentParser(
        description="Snips NLU command line interface",
        prog="python -m snips_nlu", formatter_class=Formatter)
    arg_parser.add_argument("-v", "--version", action="store_true",
                            help="Print package version")
    subparsers = arg_parser.add_subparsers(
        title="available commands", metavar="command [options ...]")
    add_generate_dataset_subparser(subparsers)
    add_train_parser(subparsers)
    add_parse_parser(subparsers)
    add_download_parser(subparsers)
    add_download_all_languages_parser(subparsers)
    add_download_entity_parser(subparsers)
    add_download_language_entities_parser(subparsers)
    add_link_parser(subparsers)
    add_cross_val_metrics_parser(subparsers)
    add_train_test_metrics_parser(subparsers)
    add_version_parser(subparsers)
    add_model_version_parser(subparsers)
    return arg_parser


def main():
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        arg_parser.print_help()
        exit(1)
