from snips_nlu.constants import PACKAGE_PATH

VERSION_FILE_NAME = "__version__"

with (PACKAGE_PATH / VERSION_FILE_NAME).open() as f:
    __version__ = f.readline().strip()

__model_version__ = "0.15.0"
