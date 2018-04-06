import io
import os

from snips_nlu.constants import PACKAGE_PATH

VERSION_FILE_NAME = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION_FILE_NAME)) as f:
    __version__ = f.readline().strip()

__model_version__ = "0.14.0"
