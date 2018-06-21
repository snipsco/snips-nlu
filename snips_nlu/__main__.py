from __future__ import unicode_literals

import sys

import plac

from snips_nlu.cli import (
    cross_val_metrics, train_test_metrics, generate_dataset, download, link)
from snips_nlu.cli.inference import parse
from snips_nlu.cli.training import train
from snips_nlu.cli.utils import prints


def main():
    commands = {
        "train": train,
        "parse": parse,
        "download": download,
        "link": link,
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
