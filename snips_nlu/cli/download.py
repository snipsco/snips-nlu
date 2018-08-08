from __future__ import print_function, unicode_literals

import json
import os
import subprocess
import sys

import plac
from snips_nlu_ontology import get_all_languages

from snips_nlu import __about__
from snips_nlu.cli.link import link
from snips_nlu.cli.utils import PrettyPrintLevel, get_json, pretty_print
from snips_nlu.constants import DATA_PATH
from snips_nlu.utils import get_package_path


# inspired from
# https://github.com/explosion/spaCy/blob/master/spacy/cli/download.py

@plac.annotations(
    resource_name=("Name of the language resources to download. Can be "
                   "either a shortcut, like 'en', or the full name of the "
                   "resources like 'snips_nlu_en'", "positional", None, str),
    direct=("force direct download. Needs resource name with version and "
            "won't perform compatibility check", "flag", "d", bool),
    pip_args=("Additional arguments to be passed to `pip install` when "
              "installing the resource"))
def download(resource_name, direct=False,
             *pip_args):  # pylint:disable=keyword-arg-before-vararg
    """Download compatible resources for the specified language"""
    if direct:
        dl = _download_resources(
            '{r}/{r}.tar.gz#egg={r}'.format(r=resource_name), pip_args)
        if dl != 0:
            sys.exit(dl)
    else:
        resource_name = resource_name.lower()
        shortcuts = get_json(__about__.__shortcuts__, "Resource shortcuts")
        full_resource_name = shortcuts.get(resource_name, resource_name)
        compatibility = _get_compatibility()
        version = _get_resources_version(full_resource_name, compatibility)
        dl = _download_resources('{r}-{v}/{r}-{v}.tar.gz#egg={r}=={v}'
                                 .format(r=full_resource_name, v=version),
                                 pip_args)
        if dl != 0:
            sys.exit(dl)
        try:
            # Get package path here because link uses
            # pip.get_installed_distributions() to check if the resource is a
            # package, which fails if the resource was just installed via
            # subprocess
            package_path = get_package_path(full_resource_name)
            link(full_resource_name, resource_name, force=True,
                 resources_path=package_path)
        except:  # pylint:disable=bare-except
            pretty_print(
                "Creating a shortcut link for '{r}' didn't work.\nYou can "
                "still load the resources via its full package name: "
                "snips_nlu.load_resources('{n}')".format(r=resource_name,
                                                         n=full_resource_name),
                title="Language resources were successfully downloaded, "
                      "however linking failed.",
                level=PrettyPrintLevel.WARNING)


@plac.annotations(
    pip_args=("Additional arguments to be passed to `pip install` when "
              "installing the resources"))
def download_all_languages(*pip_args):
    """Download compatible resources for all supported languages"""
    for language in get_all_languages():
        download(language, False, *pip_args)


def _get_compatibility():
    version = __about__.__version__
    table = get_json(__about__.__compatibility__, "Compatibility table")
    compatibility = table["snips-nlu"]
    if version not in compatibility:
        pretty_print("No compatible resources found for version %s" % version,
                     title="Resources compatibility error", exits=1,
                     level=PrettyPrintLevel.ERROR)
    return compatibility[version]


def _get_resources_version(resource_name, compatibility):
    if resource_name not in compatibility:
        pretty_print("No resources found for '%s'" % resource_name,
                     title="Resources compatibility error", exits=1,
                     level=PrettyPrintLevel.ERROR)
    return compatibility[resource_name][0]


def _download_resources(filename, user_pip_args=None):
    download_url = __about__.__download_url__ + '/' + filename
    pip_args = ['--no-cache-dir', '--no-deps']
    if user_pip_args:
        pip_args.extend(user_pip_args)
    cmd = [sys.executable, '-m', 'pip', 'install'] + pip_args + [download_url]
    return subprocess.call(cmd, env=os.environ.copy())


def _get_installed_languages():
    languages = set()
    for directory in DATA_PATH.iterdir():
        if not directory.is_dir():
            continue
        with (directory / "metadata.json").open(encoding="utf8") as f:
            metadata = json.load(f)
        languages.add(metadata["language"])
    return languages
