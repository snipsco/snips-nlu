from __future__ import print_function, unicode_literals

import plac


@plac.annotations(
    language=("Language of the assistant", "positional", None, str),
    yaml_files=("List of intent and entity yaml files", "positional", None,
                str, None, "filename"))
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
    from snips_nlu.common.utils import unicode_string, json_string
    from snips_nlu.dataset import Dataset

    language = unicode_string(language)
    dataset = Dataset.from_yaml_files(language, list(yaml_files))
    print(json_string(dataset.json, indent=2, sort_keys=True))
