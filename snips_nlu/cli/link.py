from __future__ import print_function, unicode_literals

from pathlib import Path

import plac

from snips_nlu.cli.compatibility import create_symlink
from snips_nlu.cli.utils import prints
from snips_nlu.constants import DATA_PATH
from snips_nlu.utils import is_package, get_package_path


# inspired from
# https://github.com/explosion/spaCy/blob/master/spacy/cli/link.py

@plac.annotations(
    origin=("package name or local path to model", "positional", None, str),
    link_name=("name of shortcut link to create", "positional", None, str),
    force=("force overwriting of existing link", "flag", "f", bool))
def link(origin, link_name, force=False, resources_path=None):
    """
    Create a symlink for language resources within the snips_nlu/data
    directory. Accepts either the name of a pip package, or the local path to
    the resources data directory.

    Linking resources allows loading them via
    snips_nlu.load_resources(link_name).
    """
    if is_package(origin):
        resources_path = get_package_path(origin)
    else:
        resources_path = Path(origin) if resources_path is None \
            else Path(resources_path)
    if not resources_path.exists():
        raise OSError("%s not found" % str(resources_path))
    link_path = DATA_PATH / link_name
    if link_path.is_symlink() and not force:
        raise OSError("Symlink already exists: %s" % str(link_path))
    elif link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise OSError("Symlink cannot be overwritten: %s" % str(link_path))
    create_symlink(link_path, resources_path)
    prints("%s --> %s" % (str(resources_path), str(link_path)),
           "You can now load the resources via snips_nlu.load_resources('%s')"
           % link_name, title="Linking successful")
