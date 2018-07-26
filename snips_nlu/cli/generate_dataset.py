from __future__ import print_function, unicode_literals

import json

import plac

from snips_nlu.cli.dataset.assistant_dataset import AssistantDataset


@plac.annotations(
    language=("Language of the assistant", "positional", None, str),
    files=("List of intent and entity files", "positional", None, str, None,
           "filename"))
def generate_dataset(language, *files):
    """Create a Snips NLU dataset from text friendly files"""
    dataset = AssistantDataset.from_files(language, list(files))
    print(json.dumps(dataset.json, indent=2, sort_keys=True))
