import os
import unittest

from pylint.lint import Run

from snips_nlu.constants import PACKAGE_PATH, ROOT_PATH

RCFILEPATH = os.path.join(ROOT_PATH, "tools", "pylintrc")


class TestLinting(unittest.TestCase):
    def test_linting(self):
        args = ["--output-format", "parseable", "--rcfile", RCFILEPATH]
        args += all_python_files()

        run = Run(args, exit=False)
        self.assertEqual(0, run.linter.msg_status)


def all_python_files():
    files = []
    for dirpath, _, filenames in os.walk(PACKAGE_PATH):
        if "snips-nlu-resources" in dirpath:
            continue
        files += [
            os.sep.join([dirpath, f]) for f in filenames if f.endswith(".py")
        ]

    return files
