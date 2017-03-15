import unittest
import os
from snips_nlu.tests.utils import TEST_PATH
from ..nlu_engine.nlu_engine import SnipsNLUEngine


class TestNLUEngine(unittest.TestCase):
    def test_should_load_nlu_engine(self):
        assistant_path = os.path.join(TEST_PATH, "resources/test_assistant")
        engine = SnipsNLUEngine.load(assistant_path)
        self.assertIsNone(engine.builtin_intent_parser)
        self.assertIsNotNone(engine.custom_intent_parser)
