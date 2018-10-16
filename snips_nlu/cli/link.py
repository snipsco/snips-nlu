from __future__ import print_function, unicode_literals

from pathlib import Path

import plac

from snips_nlu.cli.compatibility import create_symlink
from snips_nlu.cli.utils import PrettyPrintLevel, pretty_print
from snips_nlu.constants import DATA_PATH
from snips_nlu.resources import get_resources_sub_directory
from snips_nlu.utils import get_package_path, is_package


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
    link_path, resources_dir = link_resources(origin, link_name, force,
                                              resources_path)
    pretty_print("%s --> %s" % (str(resources_dir), str(link_path)),
                 title="Linking successful", level=PrettyPrintLevel.SUCCESS)


def link_resources(origin, link_name, force, resources_path):
    if is_package(origin):
        resources_path = get_package_path(origin)
    else:
        resources_path = Path(origin) if resources_path is None \
            else Path(resources_path)
    if not resources_path.exists():
        raise OSError("%s not found" % str(resources_path))
    link_path = DATA_PATH / str(link_name)
    if link_path.is_symlink() and not force:
        raise OSError("Symlink already exists: %s" % str(link_path))
    elif link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise OSError("Symlink cannot be overwritten: %s" % str(link_path))
    resources_sub_dir = get_resources_sub_directory(resources_path)
    create_symlink(link_path, resources_sub_dir)
    return link_path, resources_sub_dir
