from __future__ import print_function, unicode_literals


def add_link_parser(subparsers):
    subparser = subparsers.add_parser(
        "link", help="Manually link downloaded resources")
    subparser.add_argument("origin", type=str,
                           help="Package name or local path to model")
    subparser.add_argument("link_name", type=str,
                           help="Name of the symbolic link which will be "
                                "created")
    subparser.add_argument("-f", "--force", action="store_true",
                           help="Force overwriting of existing link")
    subparser.set_defaults(func=_link)
    return subparser


def _link(args_namespace):
    return link(args_namespace.origin, args_namespace.link_name,
                args_namespace.force)


# inspired from
# https://github.com/explosion/spaCy/blob/master/spacy/cli/link.py

def link(origin, link_name, force=False, resources_path=None):
    """
    Create a symlink for language resources within the snips_nlu/data
    directory. Accepts either the name of a pip package, or the local path to
    the resources data directory.

    Linking resources allows loading them via
    snips_nlu.load_resources(link_name).
    """
    from snips_nlu.cli.utils import PrettyPrintLevel, pretty_print

    link_path, resources_dir = link_resources(origin, link_name, force,
                                              resources_path)
    pretty_print("%s --> %s" % (str(resources_dir), str(link_path)),
                 title="Linking successful", level=PrettyPrintLevel.SUCCESS)


def link_resources(origin, link_name, force, resources_path):
    from pathlib import Path
    from snips_nlu.cli.compatibility import create_symlink
    from snips_nlu.common.utils import get_package_path, is_package
    from snips_nlu.constants import DATA_PATH
    from snips_nlu.resources import get_resources_sub_directory

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
