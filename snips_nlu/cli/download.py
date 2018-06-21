from __future__ import print_function, unicode_literals

import os
import subprocess
import sys

import plac

from snips_nlu import __about__
from snips_nlu.cli.link import link
from snips_nlu.cli.utils import get_json, prints
from snips_nlu.utils import get_package_path


# inspired from
# https://github.com/explosion/spaCy/blob/master/spacy/cli/download.py

@plac.annotations(
    resource_name=("Name of the language resources to download. Can be "
                   "either a shortcut, like 'en', or the full name of the "
                   "resources like 'snips_nlu_en'", "positional", None, str),
    pip_args=("Additional arguments to be passed to `pip install` when "
              "installing the model"))
def download(resource_name, *pip_args):
    """Download compatible resources for the specified language"""
    resource_name = resource_name.lower()
    shortcuts = get_json(__about__.__shortcuts__, "Resource shortcuts")
    full_resource_name = shortcuts.get(resource_name, resource_name)
    compatibility = _get_compatibility()
    version = _get_resources_version(full_resource_name, compatibility)
    dl = _download_model('{r}-{v}/{r}-{v}.tar.gz#egg={r}=={v}'
                         .format(r=full_resource_name, v=version), pip_args)
    if dl != 0:
        sys.exit(dl)
    try:
        # Get package path here because link uses
        # pip.get_installed_distributions() to check if model is a
        # package, which fails if model was just installed via
        # subprocess
        package_path = get_package_path(full_resource_name)
        link(full_resource_name, resource_name, force=True,
             resources_path=package_path)
    except:  # pylint:disable=bare-except
        prints(
            "Creating a shortcut link for '{r}' didn't work, but you can "
            "still load the resources via its full package name: "
            "snips_nlu.load_resources('{n}')".format(r=resource_name,
                                                     n=full_resource_name),
            title="Language resources were successfully downloaded, however "
                  "linking failed.")


def _get_compatibility():
    version = __about__.__version__
    table = get_json(__about__.__compatibility__, "Compatibility table")
    compatibility = table["snips-nlu"]
    if version not in compatibility:
        prints("No compatible resources found for version %s" % version,
               title="Resources compatibility error", exits=1)
    return compatibility[version]


def _get_resources_version(resource_name, compatibility):
    if resource_name not in compatibility:
        prints("No resources found for '%s'" % resource_name,
               title="Resources compatibility error", exits=1)
    return compatibility[resource_name][0]


def _download_model(filename, user_pip_args=None):
    download_url = __about__.__download_url__ + '/' + filename
    pip_args = ['--no-cache-dir', '--no-deps']
    if user_pip_args:
        pip_args.extend(user_pip_args)
    cmd = [sys.executable, '-m', 'pip', 'install'] + pip_args + [download_url]
    return subprocess.call(cmd, env=os.environ.copy())
