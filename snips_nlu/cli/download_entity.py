from __future__ import print_function, unicode_literals

import sys

import plac

from future.builtins import str
from snips_nlu_ontology import (
    get_builtin_entity_shortname, get_supported_gazetteer_entities)

from snips_nlu import __about__
from snips_nlu.cli.download import download_from_resource_name
from snips_nlu.cli.link import link_resources
from snips_nlu.cli.utils import (
    PrettyPrintLevel, check_resources_alias, get_compatibility, get_json,
    get_resources_version, install_remote_package, pretty_print)
from snips_nlu.utils import get_package_path


# inspired from
# https://github.com/explosion/spaCy/blob/master/spacy/cli/download.py

@plac.annotations(
    entity_name=("Name of the builtin entity to download, e.g. "
                 "snips/musicArtist", "positional", None, str),
    language=("Language of the builtin entity", "positional", None, str),
    pip_args=("Additional arguments to be passed to `pip install` when "
              "installing the builtin entity package"))
# pylint: disable=keyword-arg-before-vararg
def download_builtin_entity(entity_name, language, *pip_args):
    """Download compatible language or gazetteer entity resources"""
    download_from_resource_name(language, pip_args, verbose=False)

    shortcuts = get_json(__about__.__shortcuts__, "Resource shortcuts")
    check_resources_alias(entity_name, shortcuts)

    compatibility = get_compatibility()
    resource_name_lower = entity_name.lower()
    long_resource_name = shortcuts.get(resource_name_lower,
                                       resource_name_lower)

    _download_and_link_entity(
        long_resource_name, entity_name, language, compatibility, pip_args)


@plac.annotations(
    language=("Language of the builtin entity", "positional", None, str),
    pip_args=("Additional arguments to be passed to `pip install` when "
              "installing the builtin entity package"))
# pylint: disable=keyword-arg-before-vararg
def download_language_builtin_entities(language, *pip_args):
    """Download all gazetteer entity resources for a given language as well as
    basic language resources for this language"""
    download_from_resource_name(language, pip_args, verbose=False)

    shortcuts = get_json(__about__.__shortcuts__, "Resource shortcuts")
    for entity_name in get_supported_gazetteer_entities(language):
        check_resources_alias(entity_name, shortcuts)

        compatibility = get_compatibility()
        resource_name_lower = entity_name.lower()
        long_resource_name = shortcuts.get(resource_name_lower,
                                           resource_name_lower)

        _download_and_link_entity(
            long_resource_name, entity_name, language, compatibility, pip_args)


# pylint: enable=keyword-arg-before-vararg


def _download_and_link_entity(long_resource_name, entity_name, language,
                              compatibility, pip_args):
    full_resource_name = long_resource_name + "_" + language
    version = get_resources_version(full_resource_name, entity_name,
                                    compatibility)
    entity_alias = get_builtin_entity_shortname(entity_name).lower()
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
    if not version.startswith("v"):
        version = "v" + version
    return "/".join(
        [__about__.__entities_download_url__, language, entity_alias, version])
