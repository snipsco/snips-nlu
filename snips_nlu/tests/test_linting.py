import os
import unittest

import time
from pylint.lint import Run

import snips_nlu

RCFILEPATH = os.path.join(snips_nlu.ROOT_PATH, "tools", "pylintrc")


class TestLinting(unittest.TestCase):
    def test_linting(self):
        args = ["--output-format", "parseable", "--rcfile", RCFILEPATH]
        args += all_python_files()

        run = Run(args, exit=False)
        time.sleep(10)
        self.assertEqual(0, run.linter.msg_status)


def all_python_files():
    files = []
    for dirpath, _, filenames in os.walk(snips_nlu.PACKAGE_PATH):
        if "snips-nlu-resources" in dirpath:
            continue
        files += [
            os.sep.join([dirpath, f]) for f in filenames if f.endswith(".py")
        ]

    return files
