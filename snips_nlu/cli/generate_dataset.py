from __future__ import print_function, unicode_literals

import plac

from snips_nlu.dataset import Dataset
from snips_nlu.common.utils import unicode_string, json_string


@plac.annotations(
    language=("Language of the assistant", "positional", None, str),
    files=("List of intent and entity files", "positional", None, str, None,
           "filename"))
def generate_dataset(language, *files):
    """Create a Snips NLU dataset from text friendly files"""
    language = unicode_string(language)
    if any(f.endswith(".yml") or f.endswith(".yaml") for f in files):
        dataset = Dataset.from_yaml_files(language, list(files))
    else:
        dataset = Dataset.from_files(language, list(files))
    print(json_string(dataset.json, indent=2, sort_keys=True))
