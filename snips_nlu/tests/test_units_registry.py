import unittest

from deprecation import fail_if_not_removed

from snips_nlu.intent_parser import IntentParser
from snips_nlu.pipeline.units_registry import register_processing_unit


class TestUnitsRegistry(unittest.TestCase):
    @fail_if_not_removed
    def test_should_register_processing_unit(self):
        # Given
        # pylint:disable=abstract-method
        class MyIntentParser(IntentParser):
            unit_name = "my_old_intent_parser"

            def fit(self, dataset, force_retrain):
                pass

            def parse(self, text, intents, top_n):
                pass

            def get_intents(self, text):
                pass

            def get_slots(self, text, intent):
                pass

        # pylint:enable=abstract-method

        # When
        register_processing_unit(MyIntentParser)

        # Then
        parser_name = IntentParser.registered_name(MyIntentParser)
        self.assertEqual("my_old_intent_parser", parser_name)
