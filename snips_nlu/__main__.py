from __future__ import unicode_literals

import sys

import plac

from snips_nlu.cli import cross_val_metrics, train_test_metrics, \
    generate_dataset
from snips_nlu.cli.inference import parse
from snips_nlu.cli.training import train


def prints(*texts, **kwargs):
    """Print formatted message (manual ANSI escape sequences to avoid
    dependency)
    *texts (unicode): Texts to print. Each argument is rendered as paragraph.
    **kwargs: 'title' becomes coloured headline. exits=True performs sys exit.
    """
    exits = kwargs.get('exits', None)
    title = kwargs.get('title', None)
    title = '\033[93m{}\033[0m\n'.format(title) if title else ''
    message = '\n\n'.join([text for text in texts])
    print('\n{}{}\n'.format(title, message))
    if exits is not None:
        sys.exit(exits)


def main():
    commands = {
        "train": train,
        "parse": parse,
        "generate-dataset": generate_dataset,
        "cross-val-metrics": cross_val_metrics,
        "train-test-metrics": train_test_metrics,
    }
    if len(sys.argv) == 1:
        prints(', '.join(commands), title="Available commands", exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = 'snips-nlu %s' % command
    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        prints("Available: %s" % ', '.join(commands),
               title="Unknown command: %s" % command, exits=1)


if __name__ == "__main__":
    main()
