from __future__ import print_function, unicode_literals


def main():
    import sys

    import plac

    from snips_nlu.__about__ import __version__, __model_version__
    from snips_nlu.cli import (
        cross_val_metrics, download, download_all_languages, generate_dataset,
        link, train_test_metrics)
    from snips_nlu.cli.download_entity import (
        download_builtin_entity, download_language_builtin_entities)
    from snips_nlu.cli.inference import parse
    from snips_nlu.cli.training import train
    from snips_nlu.cli.utils import PrettyPrintLevel, pretty_print

    commands = {
        "train": train,
        "parse": parse,
        "download": download,
        "download-all-languages": download_all_languages,
        "download-entity": download_builtin_entity,
        "download-language-entities": download_language_builtin_entities,
        "version": lambda: print(__version__),
        "model-version": lambda: print(__model_version__),
        "link": link,
        "generate-dataset": generate_dataset,
        "cross-val-metrics": cross_val_metrics,
        "train-test-metrics": train_test_metrics,
    }
    if len(sys.argv) == 1:
        pretty_print(', '.join(commands), title="Available commands", exits=1,
                     level=PrettyPrintLevel.INFO)
    command = sys.argv.pop(1)
    sys.argv[0] = 'snips-nlu %s' % command
    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        pretty_print("Available: %s" % ', '.join(commands),
                     title="Unknown command: %s" % command, exits=1,
                     level=PrettyPrintLevel.INFO)


if __name__ == "__main__":
    main()
