from __future__ import unicode_literals
import io
import os

import builtin_entities_ontology

from snips_nlu.resources import load_resources
from snips_nlu.utils import ROOT_PATH, PACKAGE_NAME

__model_version__ = "0.8.5"

VERSION_FILE_NAME = "__version__"

with io.open(os.path.join(ROOT_PATH, PACKAGE_NAME, VERSION_FILE_NAME)) as f:
    __version__ = f.readline().strip()

load_resources()

__builtin_entities_version__ = builtin_entities_ontology.__ontology_version__
