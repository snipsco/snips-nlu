from __future__ import print_function, unicode_literals


def add_download_entity_parser(subparsers):
    subparser = subparsers.add_parser(
        "download-entity",
        help="Download resources for a builtin gazetteer entity")
    subparser.add_argument(
        "entity_name", type=str,
        help="Name of the builtin entity to download, e.g. snips/musicArtist")
    subparser.add_argument("language", type=str,
                           help="Language of the builtin entity")
    subparser.add_argument(
        "extra_pip_args", nargs="*", type=str,
        help="Additional arguments to be passed to `pip install` when "
             "installing the builtin entity package")
    subparser.set_defaults(func=_download_builtin_entity)
    return subparser


def _download_builtin_entity(args_namespace):
    return download_builtin_entity(
        args_namespace.entity_name, args_namespace.language,
        *args_namespace.extra_pip_args)


# inspired from
# https://github.com/explosion/spaCy/blob/master/spacy/cli/download.py

def download_builtin_entity(entity_name, language, *pip_args):
    """Download compatible language or gazetteer entity resources"""
    from snips_nlu import __about__
    from snips_nlu.cli.download import download_from_resource_name
    from snips_nlu.cli.utils import (
        check_resources_alias, get_compatibility, get_json)

    download_from_resource_name(language, pip_args, verbose=False)

    shortcuts = get_json(__about__.__shortcuts__, "Resource shortcuts")
    check_resources_alias(entity_name, shortcuts)

    compatibility = get_compatibility()
    resource_name_lower = entity_name.lower()
    long_resource_name = shortcuts.get(resource_name_lower,
                                       resource_name_lower)

    _download_and_link_entity(
        long_resource_name, entity_name, language, compatibility,
        pip_args)


def add_download_language_entities_parser(subparsers):
    subparser = subparsers.add_parser(
        "download-language-entities",
        help="Download resources for all builtin gazetteer entities in a "
             "given language")
    subparser.add_argument("language", type=str,
                           help="Language of the builtin entities")
    subparser.add_argument(
        "extra_pip_args", nargs="*", type=str,
        help="Additional arguments to be passed to `pip install` when "
             "installing the builtin entities packages")
    subparser.set_defaults(func=_download_language_builtin_entities)
    return subparser


def _download_language_builtin_entities(args_namespace):
    return download_language_builtin_entities(
        args_namespace.language, *args_namespace.extra_pip_args)


def download_language_builtin_entities(language, *pip_args):
    """Download all gazetteer entity resources for a given language as well as
    basic language resources for this language"""
    from builtins import str
    from snips_nlu_parsers import get_supported_gazetteer_entities
    from snips_nlu import __about__
    from snips_nlu.cli.download import download_from_resource_name
    from snips_nlu.cli.utils import (
        check_resources_alias, get_compatibility, get_json)

    download_from_resource_name(language, pip_args, verbose=False)

    shortcuts = get_json(__about__.__shortcuts__, "Resource shortcuts")
    for entity_name in get_supported_gazetteer_entities(str(language)):
        check_resources_alias(entity_name, shortcuts)

        compatibility = get_compatibility()
        resource_name_lower = entity_name.lower()
        long_resource_name = shortcuts.get(resource_name_lower,
                                           resource_name_lower)

        _download_and_link_entity(
            long_resource_name, entity_name, language, compatibility,
            pip_args)


def _download_and_link_entity(long_resource_name, entity_name, language,
                              compatibility, pip_args):
    import sys
    from builtins import str
    from snips_nlu_parsers import get_builtin_entity_shortname
    from snips_nlu.cli.link import link_resources
    from snips_nlu.cli.utils import (
        PrettyPrintLevel, get_json, get_resources_version,
        install_remote_package, pretty_print)
    from snips_nlu.common.utils import get_package_path

    full_resource_name = long_resource_name + "_" + language
    version = get_resources_version(full_resource_name, entity_name,
                                    compatibility)
    entity_alias = get_builtin_entity_shortname(str(entity_name)).lower()
    entity_base_url = _get_entity_base_url(language, entity_alias, version)
    latest = get_json(entity_base_url + "/latest",
                      "Latest entity resources version")
    latest_url = "{b}/{n}#egg={r}=={v}".format(
        b=entity_base_url, n=latest["filename"], r=full_resource_name,
        v=latest["version"])
    exit_code = install_remote_package(latest_url, pip_args)
    if exit_code != 0:
        sys.exit(exit_code)
    try:
        # Get package path here because link uses
        # pip.get_installed_distributions() to check if the resource is a
        # package, which fails if the resource was just installed via
        # subprocess
        package_path = get_package_path(full_resource_name)
        link_alias = entity_alias + "_" + language
        link_path, resources_dir = link_resources(
            full_resource_name, link_alias, force=True,
            resources_path=package_path)
        pretty_print("%s --> %s" % (str(resources_dir), str(link_path)),
                     "You can now use the '%s' builtin entity" % entity_name,
                     title="Linking successful",
                     level=PrettyPrintLevel.SUCCESS)
    except:  # pylint:disable=bare-except
        pretty_print(
            "Creating a shortcut link for '%s' didn't work." % entity_name,
            title="The builtin entity resources were successfully downloaded, "
                  "however linking failed.",
            level=PrettyPrintLevel.ERROR)


def _get_entity_base_url(language, entity_alias, version):
    from snips_nlu import __about__

    if not version.startswith("v"):
        version = "v" + version
    return "/".join(
        [__about__.__entities_download_url__, language, entity_alias, version])
