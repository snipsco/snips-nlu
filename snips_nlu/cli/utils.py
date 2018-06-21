from __future__ import print_function, unicode_literals

import sys

import requests


def prints(*texts, **kwargs):
    """Print formatted message

    Args:
        *texts (str): Texts to print. Each argument is rendered as paragraph.
        **kwargs: 'title' becomes coloured headline. exits=True performs sys
        exit.
    """
    exits = kwargs.get("exits", None)
    title = kwargs.get("title", None)
    title = "\033[93m{title}\033[0m\n".format(title=title) if title else ""
    message = "\n\n".join([text for text in texts])
    print("\n{title}{message}\n".format(title=title, message=message))
    if exits is not None:
        sys.exit(exits)


def get_json(url, desc):
    r = requests.get(url)
    if r.status_code != 200:
        raise OSError("%s: Received status code %s when fetching the resource"
                      % (desc, r.status_code))
    return r.json()
