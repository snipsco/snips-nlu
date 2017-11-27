from __future__ import unicode_literals

import io
import os

import builtin_entities_ontology

from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.resources import load_resources
from snips_nlu.utils import ROOT_PATH, PACKAGE_NAME, PACKAGE_PATH

__model_version__ = "0.11.0"

VERSION_FILE_NAME = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION_FILE_NAME)) as f:
    __version__ = f.readline().strip()

load_resources()

__builtin_entities_version__ = builtin_entities_ontology.__ontology_version__
