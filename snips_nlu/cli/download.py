from __future__ import print_function, unicode_literals

import sys

import plac
from snips_nlu_parsers import get_all_languages

from snips_nlu import __about__
from snips_nlu.cli.link import link_resources
from snips_nlu.cli.utils import (
    PrettyPrintLevel, check_resources_alias, get_compatibility, get_json,
    get_resources_version, install_remote_package, pretty_print)
from snips_nlu.common.utils import get_package_path


# inspired from
# https://github.com/explosion/spaCy/blob/master/spacy/cli/download.py

@plac.annotations(
    resource_name=("Name of the language resources to download. Can be either "
                   "a shortcut, like 'en', or the full name of the resources "
                   "like 'snips_nlu_en'", "positional", None, str),
    direct=("force direct download. Needs resource name with version and "
            "won't perform compatibility check", "flag", "d", bool),
    pip_args=("Additional arguments to be passed to `pip install` when "
              "installing the language resources package"))
def download(resource_name, direct=False,
             *pip_args):  # pylint:disable=keyword-arg-before-vararg
    """Download compatible language resources"""
    if direct:
        components = resource_name.split("-")
        name = "".join(components[:-1])
        version = components[-1]
        url_tail = '{n}-{v}/{n}-{v}.tar.gz#egg={n}=={v}'.format(
            n=name, v=version)
        download_url = __about__.__download_url__ + '/' + url_tail
        dl = install_remote_package(download_url, pip_args)
        if dl != 0:
            sys.exit(dl)
    else:
        download_from_resource_name(resource_name, pip_args)


def download_from_resource_name(resource_name, pip_args, verbose=True):
    shortcuts = get_json(__about__.__shortcuts__, "Resource shortcuts")
    check_resources_alias(resource_name, shortcuts)
    compatibility = get_compatibility()
    resource_name = resource_name.lower()
    full_resource_name = shortcuts.get(resource_name, resource_name)
    _download_and_link(resource_name, full_resource_name, compatibility,
                       pip_args, verbose)


@plac.annotations(
    pip_args=("Additional arguments to be passed to `pip install` when "
              "installing the resources"))
def download_all_languages(*pip_args):
    """Download compatible resources for all supported languages"""
    for language in get_all_languages():
        download(language, False, *pip_args)


def _download_and_link(resource_alias, resource_fullname, compatibility,
                       pip_args, verbose):
    version = get_resources_version(resource_fullname, resource_alias,
                                    compatibility)
    url_tail = '{r}-{v}/{r}-{v}.tar.gz#egg={r}=={v}'.format(
        r=resource_fullname, v=version)
    download_url = __about__.__download_url__ + '/' + url_tail
    exit_code = install_remote_package(download_url, pip_args)
    if exit_code != 0:
        sys.exit(exit_code)
    try:
        # Get package path here because link uses
        # pip.get_installed_distributions() to check if the resource is a
        # package, which fails if the resource was just installed via
        # subprocess
        package_path = get_package_path(resource_fullname)
        link_path, resources_dir = link_resources(
            resource_fullname, resource_alias, force=True,
            resources_path=package_path)
        if verbose:
            pretty_print("%s --> %s" % (str(resources_dir), str(link_path)),
                         title="Linking successful",
                         level=PrettyPrintLevel.SUCCESS)
    except OSError as e:  # pylint:disable=bare-except
        pretty_print(
            "Creating a shortcut link for '%s' didn't work: %s"
            % (resource_alias, repr(e)),
            title="The language resources were successfully downloaded, "
                  "however linking failed.",
            level=PrettyPrintLevel.ERROR)
