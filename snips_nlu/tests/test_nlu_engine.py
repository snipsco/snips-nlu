import os
import unittest

from snips_nlu.tests.utils import TEST_PATH
from ..nlu_engine.nlu_engine import SnipsNLUEngine


class TestNLUEngine(unittest.TestCase):
    def test_should_load_nlu_engine(self):
        assistant_path = os.path.join(TEST_PATH, "resources/test_assistant")
        engine = SnipsNLUEngine.load(assistant_path)
        self.assertEqual(len(engine.custom_parsers), 1)
        self.assertEqual(len(engine.builtin_parsers), 0)
