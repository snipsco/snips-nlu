from __future__ import print_function, unicode_literals


def add_generate_dataset_subparser(subparsers):
    subparser = subparsers.add_parser(
        "generate-dataset",
        help="Generate a json dataset from intents and entities yaml files")
    subparser.add_argument("language", type=str,
                           help="Language of the dataset")
    subparser.add_argument("files", nargs="+", type=str,
                           help="List of intent and entity yaml files")
    subparser.set_defaults(func=_generate_dataset)
    return subparser


def _generate_dataset(args_namespace):
    return generate_dataset(args_namespace.language, *args_namespace.files)


def generate_dataset(language, *yaml_files):
    """Creates a Snips NLU dataset from YAML definition files

    Check :meth:`.Intent.from_yaml` and :meth:`.Entity.from_yaml` for the
    format of the YAML files.

    Args:
        language (str): language of the dataset (iso code)
        *yaml_files: list of intent and entity definition files in YAML format.

    Returns:
        None. The json dataset output is printed out on stdout.
    """
    from snips_nlu.dataset import Dataset
    from snips_nlu.common.utils import unicode_string, json_string

    language = unicode_string(language)
    dataset = Dataset.from_yaml_files(language, list(yaml_files))
    print(json_string(dataset.json, indent=2, sort_keys=True))
