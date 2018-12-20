import io

from snips_nlu import SnipsNLUEngine, NLUEngineConfig
from snips_nlu.constants import START, END
from snips_nlu.dataset import Dataset
from snips_nlu.pipeline.configs import ProbabilisticIntentParserConfig
from snips_nlu.result import unresolved_slot, custom_slot
from snips_nlu.slot_filler.keyword_slot_filler import KeywordSlotFiller
from snips_nlu.tests.utils import FixtureTest


class TestKeywordSlotFiller(FixtureTest):
    def test_should_get_slots_with_keywords_slot_filler(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: SetLightColor
utterances:
- set the light to [color](blue) in the [room](kitchen)
- please make the lights [color](red) in the [room](bathroom)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        intent = "SetLightColor"
        slot_filler = KeywordSlotFiller().fit(dataset, intent)

        # When
        slots = slot_filler.get_slots("I want red lights in the kitchen now")

        # Then
        expected_slots = [
            unresolved_slot(match_range={START: 7, END: 10},
                            value="red",
                            entity="color",
                            slot_name="color"),
            unresolved_slot(match_range={START: 25, END: 32},
                            value="kitchen",
                            entity="room",
                            slot_name="room")
        ]
        self.assertListEqual(slots, expected_slots)

    def test_keywords_slot_filler_should_work_in_engine(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: SetLightColor
utterances:
- set the light to [color](blue) in the [room](kitchen)
- please make the lights [color](red) in the [room](bathroom)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        intent = "SetLightColor"
        slot_filler_config = {
            "unit_name": "keyword_slot_filler",
            "lowercase": True
        }
        parser_config = ProbabilisticIntentParserConfig(
            slot_filler_config=slot_filler_config)
        engine_config = NLUEngineConfig([parser_config])
        engine = SnipsNLUEngine(engine_config).fit(dataset, intent)
        text = "I want Red lights in the kitchen now"

        # When
        res = engine.parse(text)

        # Then
        expected_slots = [
            custom_slot(unresolved_slot(match_range={START: 7, END: 10},
                                        value="Red",
                                        entity="color",
                                        slot_name="color"), "red"),
            custom_slot(unresolved_slot(match_range={START: 25, END: 32},
                                        value="kitchen",
                                        entity="room",
                                        slot_name="room"))
        ]
        self.assertListEqual(expected_slots, res["slots"])

    def test_engine_with_keyword_slot_filler_should_be_serializable(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: SetLightColor
utterances:
- set the light to [color](blue) in the [room](kitchen)
- please make the lights [color](red) in the [room](bathroom)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        intent = "SetLightColor"
        slot_filler_config = {
            "unit_name": "keyword_slot_filler",
            "lowercase": True
        }
        parser_config = ProbabilisticIntentParserConfig(
            slot_filler_config=slot_filler_config)
        engine_config = NLUEngineConfig([parser_config])
        engine = SnipsNLUEngine(engine_config).fit(dataset, intent)
        engine.persist(self.tmp_file_path)
        text = "I want Red lights in the kitchen now"

        # When
        loaded_engine = SnipsNLUEngine.from_path(self.tmp_file_path)
        res = loaded_engine.parse(text)

        # Then
        expected_slots = [
            custom_slot(unresolved_slot(match_range={START: 7, END: 10},
                                        value="Red",
                                        entity="color",
                                        slot_name="color"), "red"),
            custom_slot(unresolved_slot(match_range={START: 25, END: 32},
                                        value="kitchen",
                                        entity="room",
                                        slot_name="room"))
        ]
        self.assertListEqual(expected_slots, res["slots"])
