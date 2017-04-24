import io
import os

from duckling import core

from snips_nlu.utils import ROOT_PATH

core.load()

VERSION_FILE_NAME = "__version__"

with io.open(os.path.join(ROOT_PATH, VERSION_FILE_NAME)) as f:
    __version__ = f.readline().strip()
