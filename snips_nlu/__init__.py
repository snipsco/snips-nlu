import io
import os

from duckling import core

from snips_nlu.resources import load_resources
from snips_nlu.utils import ROOT_PATH, PACKAGE_NAME

core.load()

VERSION_FILE_NAME = "__version__"

with io.open(os.path.join(ROOT_PATH, PACKAGE_NAME, VERSION_FILE_NAME)) as f:
    __version__ = f.readline().strip()

load_resources()
