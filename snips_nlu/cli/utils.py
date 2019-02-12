from __future__ import print_function, unicode_literals

import logging
import os
import subprocess
import sys
from enum import Enum, unique

import requests
from semantic_version import Version

import snips_nlu
from snips_nlu import __about__


@unique
class PrettyPrintLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    SUCCESS = 3


FMT = "[%(levelname)s][%(asctime)s.%(msecs)03d][%(name)s]: " \
      "%(message)s"
DATE_FMT = "%H:%M:%S"


def pretty_print(*texts, **kwargs):
    """Print formatted message

    Args:
        *texts (str): Texts to print. Each argument is rendered as paragraph.
        **kwargs: 'title' becomes coloured headline. exits=True performs sys
        exit.
    """
    exits = kwargs.get("exits")
    title = kwargs.get("title")
    level = kwargs.get("level", PrettyPrintLevel.INFO)
    title_color = _color_from_level(level)
    if title:
        title = "\033[{color}m{title}\033[0m\n".format(title=title,
                                                       color=title_color)
    else:
        title = ""
    message = "\n\n".join([text for text in texts])
    print("\n{title}{message}\n".format(title=title, message=message))
    if exits is not None:
        sys.exit(exits)


def _color_from_level(level):
    if level == PrettyPrintLevel.INFO:
        return "92"
    if level == PrettyPrintLevel.WARNING:
        return "93"
    if level == PrettyPrintLevel.ERROR:
        return "91"
    if level == PrettyPrintLevel.SUCCESS:
        return "92"
    else:
        raise ValueError("Unknown PrettyPrintLevel: %s" % level)


def get_json(url, desc):
    r = requests.get(url)
    if r.status_code != 200:
        raise OSError("%s: Received status code %s when fetching the resource"
                      % (desc, r.status_code))
    return r.json()


def get_compatibility():
    version = __about__.__version__
    semver_version = Version(version)
    minor_version = "%d.%d" % (semver_version.major, semver_version.minor)
    table = get_json(__about__.__compatibility__, "Compatibility table")
    nlu_table = table["snips-nlu"]
    compatibility = nlu_table.get(version, nlu_table.get(minor_version))
    if compatibility is None:
        pretty_print("No compatible resources found for version %s" % version,
                     title="Resources compatibility error", exits=1,
                     level=PrettyPrintLevel.ERROR)
    return compatibility


def get_resources_version(resource_fullname, resource_alias, compatibility):
    if resource_fullname not in compatibility:
        pretty_print("No compatible resources found for '%s'" % resource_alias,
                     title="Resources compatibility error", exits=1,
                     level=PrettyPrintLevel.ERROR)
    return compatibility[resource_fullname][0]


def install_remote_package(download_url, user_pip_args=None):
    pip_args = ['--no-cache-dir', '--no-deps']
    if user_pip_args:
        pip_args.extend(user_pip_args)
    cmd = [sys.executable, '-m', 'pip', 'install'] + pip_args + [download_url]
    return subprocess.call(cmd, env=os.environ.copy())


def check_resources_alias(resource_name, shortcuts):
    available_aliases = set(shortcuts)
    if resource_name.lower() not in available_aliases:
        aliases = ", ".join(sorted(available_aliases))
        pretty_print(
            "No resources found for {r}, available resource aliases are "
            "(case insensitive):\n{a}".format(r=resource_name, a=aliases),
            title="Unknown language resources", exits=1,
            level=PrettyPrintLevel.ERROR)


def set_nlu_logger(level=logging.INFO):
    logger = logging.getLogger(snips_nlu.__name__)
    logger.setLevel(level)

    formatter = logging.Formatter(FMT, DATE_FMT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.addHandler(handler)
