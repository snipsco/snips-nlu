import unittest

from snips_nlu.tests.utils import SAMPLE_DATASET
from ..nlu_engine.nlu_engine import SnipsNLUEngine
from ..result import result, parsed_entity, intent_classification_result


class TestNLUEngine(unittest.TestCase):
    def test_should_load_nlu_engine_from_dict(self):
        nlu_engine_dict = dict()
        engine = SnipsNLUEngine.load_from_dict(nlu_engine_dict)
        self.assertEqual(len(engine.custom_parsers), 0)
        self.assertEqual(len(engine.builtin_parsers), 0)

    def test_should_fit_nlu_engine(self):
        # Given
        dataset = SAMPLE_DATASET
        engine = SnipsNLUEngine.load_from_dict(dict())

        # When
        engine.fit(dataset)

        # Then
        self.assertEqual(len(engine.custom_parsers), 2)
        self.assertEqual(len(engine.builtin_parsers), 0)

    def test_should_save_nlu_engine(self):
        # Given
        dataset = SAMPLE_DATASET
        engine = SnipsNLUEngine.load_from_dict(dict()).fit(dataset)
        pkl_str = engine.save_to_pickle_string()

        # When
        new_engine = SnipsNLUEngine.load_from_pickle_and_bytearray(pkl_str,
                                                                   None)

        # Then
        self.assertEqual(engine, new_engine)

    def test_nlu_engine_should_parse_text(self):
        # Given
        dataset = SAMPLE_DATASET
        engine = SnipsNLUEngine.load_from_dict(dict()).fit(dataset)

        # When
        text = "this is a dummy_a query with another dummy_c"
        parse = engine.parse(text)

        # Then
        expected_entities = [
            parsed_entity(match_range=(10, 17), value="dummy_a",
                          entity="dummy_entity_1", slot_name="dummy_slot_name"),
            parsed_entity(match_range=(37, 44), value="dummy_c",
                          entity="dummy_entity_2", slot_name="dummy_slot_name2")
        ]
        expected_proba = (len("dummy_a") + len("dummy_c")) / float(len(text))
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1",
            probability=expected_proba)
        expected_parse = result(text=text, parsed_intent=expected_intent,
                                parsed_entities=expected_entities)
        self.assertEqual(parse, expected_parse)
