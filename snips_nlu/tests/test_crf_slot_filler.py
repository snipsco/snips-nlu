# coding=utf-8
from __future__ import unicode_literals

import io
from builtins import range
from pathlib import Path

from mock import MagicMock
from sklearn_crfsuite import CRF

from snips_nlu.constants import (
    DATA, END, ENTITY, LANGUAGE_EN, SLOT_NAME, START, TEXT)
from snips_nlu.dataset import Dataset
from snips_nlu.entity_parser import CustomEntityParserUsage
from snips_nlu.exceptions import NotTrained
from snips_nlu.pipeline.configs import CRFSlotFillerConfig
from snips_nlu.preprocessing import tokenize
from snips_nlu.result import unresolved_slot
from snips_nlu.slot_filler.crf_slot_filler import CRFSlotFiller, _ensure_safe
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_factory import (
    IsDigitFactory, NgramFactory, ShapeNgramFactory)
from snips_nlu.tests.utils import FixtureTest, TEST_PATH


class TestCRFSlotFiller(FixtureTest):
    def test_should_get_slots(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me [number_of_cups:snips/number](five) cups of tea
- please I want [number_of_cups](two) cups of tea""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        config = CRFSlotFillerConfig()
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(config, **shared)
        intent = "MakeTea"
        slot_filler.fit(dataset, intent)

        # When
        slots = slot_filler.get_slots("make me two cups of tea")

        # Then
        expected_slots = [
            unresolved_slot(match_range={START: 8, END: 11},
                            value='two',
                            entity='snips/number',
                            slot_name='number_of_cups')]
        self.assertListEqual(slots, expected_slots)

    def test_should_get_builtin_slots(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: GetWeather
utterances:
- what is the weather [datetime:snips/datetime](at 9pm)
- what's the weather in [location:weather_location](berlin)
- What's the weather in [location](tokyo) [datetime](this weekend)?
- Can you tell me the weather [datetime] please ?
- what is the weather forecast [datetime] in [location](paris)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        config = CRFSlotFillerConfig()
        intent = "GetWeather"
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(config, **shared)
        slot_filler.fit(dataset, intent)

        # When
        slots = slot_filler.get_slots("Give me the weather at 9pm in Paris")

        # Then
        expected_slots = [
            unresolved_slot(match_range={START: 20, END: 26},
                            value='at 9pm',
                            entity='snips/datetime',
                            slot_name='datetime'),
            unresolved_slot(match_range={START: 30, END: 35},
                            value='Paris',
                            entity='weather_location',
                            slot_name='location')
        ]
        self.assertListEqual(expected_slots, slots)

    def test_should_get_sub_builtin_slots(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: PlanBreak
utterances:
- 'I want to leave from [start:snips/datetime](tomorrow) until 
  [end:snips/datetime](next thursday)'
- find me something from [start](9am) to [end](12pm)
- I need a break from [start](2pm) until [end](4pm)
- Can you suggest something from [start](april 4th) until [end](april 6th) ?
- find an activity from [start](6pm) to [end](8pm)
- Book me a trip from [start](this friday) to [end](next tuesday)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        config = CRFSlotFillerConfig()
        intent = "PlanBreak"
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(config, **shared)
        slot_filler.fit(dataset, intent)

        # When
        slots = slot_filler.get_slots("Find me a plan from 5pm to 6pm")

        # Then
        expected_slots = [
            unresolved_slot(match_range={START: 20, END: 23},
                            value="5pm",
                            entity="snips/datetime",
                            slot_name="start"),
            unresolved_slot(match_range={START: 27, END: 30},
                            value="6pm",
                            entity="snips/datetime",
                            slot_name="end")
        ]
        self.assertListEqual(expected_slots, slots)

    def test_should_not_use_crf_when_dataset_with_no_slots(self):
        # Given
        dataset = {
            "language": "en",
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "This is an utterance without "
                                            "slots"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {}
        }
        slot_filler = CRFSlotFiller(**self.get_shared_data(dataset))
        mock_compute_features = MagicMock()
        slot_filler.compute_features = mock_compute_features

        # When
        slot_filler.fit(dataset, "intent1")
        slots = slot_filler.get_slots("This is an utterance without slots")

        # Then
        mock_compute_features.assert_not_called()
        self.assertListEqual([], slots)

    def test_should_compute_sequence_probability_when_no_slots(self):
        # Given
        dataset = {
            "language": "en",
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "This is an utterance without "
                                            "slots"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {}
        }
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared).fit(dataset, "intent1")
        tokens = tokenize("hello world foo bar", "en")

        # When
        res1 = slot_filler.get_sequence_probability(
            tokens, ["O", "O", "O", "O"])
        res2 = slot_filler.get_sequence_probability(
            tokens, ["O", "O", "B-location", "O"])

        # Then
        self.assertEqual(1.0, res1)
        self.assertEqual(0.0, res2)

    def test_should_parse_naughty_strings(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: my_intent
utterances:
- this is [entity1](my first entity)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        naughty_strings_path = TEST_PATH / "resources" / "naughty_strings.txt"
        with naughty_strings_path.open(encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        # When
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared).fit(dataset, "my_intent")

        # Then
        for s in naughty_strings:
            with self.fail_if_exception("Naughty string crashes"):
                slot_filler.get_slots(s)

    def test_should_not_get_slots_when_not_fitted(self):
        # Given
        slot_filler = CRFSlotFiller()

        # When / Then
        self.assertFalse(slot_filler.fitted)
        with self.assertRaises(NotTrained):
            slot_filler.get_slots("foobar")

    def test_should_not_get_sequence_probability_when_not_fitted(self):
        # Given
        slot_filler = CRFSlotFiller()

        # When / Then
        with self.assertRaises(NotTrained):
            slot_filler.get_sequence_probability(tokens=[], labels=[])

    def test_should_not_log_weights_when_not_fitted(self):
        # Given
        slot_filler = CRFSlotFiller()

        # When / Then
        with self.assertRaises(NotTrained):
            slot_filler.log_weights()

    def test_should_fit_with_naughty_strings_no_tags(self):
        # Given
        naughty_strings_path = TEST_PATH / "resources" / "naughty_strings.txt"
        with naughty_strings_path.open(encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        utterances = [{DATA: [{TEXT: naughty_string}]} for naughty_string in
                      naughty_strings]

        # When
        naughty_dataset = {
            "intents": {
                "naughty_intent": {
                    "utterances": utterances
                }
            },
            "entities": dict(),
            "language": "en",
        }

        # Then
        with self.fail_if_exception("Naughty string crashes"):
            shared = self.get_shared_data(naughty_dataset)
            CRFSlotFiller(**shared).fit(naughty_dataset, "naughty_intent")

    def test_should_fit_and_parse_with_non_ascii_tags(self):
        # Given
        inputs = ("string%s" % i for i in range(10))
        utterances = [{
            DATA: [{
                TEXT: string,
                ENTITY: "non_ascìi_entïty",
                SLOT_NAME: "non_ascìi_slöt"
            }]
        } for string in inputs]

        # When
        naughty_dataset = {
            "intents": {
                "naughty_intent": {
                    "utterances": utterances
                }
            },
            "entities": {
                "non_ascìi_entïty": {
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "data": [],
                    "matching_strictness": 1.0
                }
            },
            "language": "en",
        }

        # Then
        with self.fail_if_exception("Naughty string make NLU crash"):
            shared = self.get_shared_data(naughty_dataset)
            slot_filler = CRFSlotFiller(**shared)
            slot_filler.fit(naughty_dataset, "naughty_intent")
            slots = slot_filler.get_slots("string0")
            expected_slot = {
                "entity": "non_ascìi_entïty",
                "range": {
                    "start": 0,
                    "end": 7
                },
                "slotName": u"non_ascìi_slöt",
                "value": u"string0"
            }
            self.assertListEqual([expected_slot], slots)

    def test_should_get_slots_after_deserialization(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me [number_of_cups:snips/number](one) cup of tea
- i want [number_of_cups] cups of tea please
- can you prepare [number_of_cups] cups of tea ?""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        config = CRFSlotFillerConfig()
        intent = "MakeTea"
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(config, **shared)
        slot_filler.fit(dataset, intent)
        slot_filler.persist(self.tmp_file_path)

        deserialized_slot_filler = CRFSlotFiller.from_path(
            self.tmp_file_path, **shared)

        # When
        slots = deserialized_slot_filler.get_slots("make me two cups of tea")

        # Then
        expected_slots = [
            unresolved_slot(match_range={START: 8, END: 11},
                            value='two',
                            entity='snips/number',
                            slot_name='number_of_cups')]
        self.assertListEqual(expected_slots, slots)

    def test_should_be_serializable_before_fit(self):
        # Given
        features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(
            tagging_scheme=TaggingScheme.BILOU,
            feature_factory_configs=features_factories)

        slot_filler = CRFSlotFiller(config)

        # When
        slot_filler.persist(self.tmp_file_path)

        # Then
        metadata_path = self.tmp_file_path / "metadata.json"
        self.assertJsonContent(metadata_path, {"unit_name": "crf_slot_filler"})

        expected_slot_filler_dict = {
            "crf_model_file": None,
            "language_code": None,
            "config": config.to_dict(),
            "intent": None,
            "slot_name_mapping": None,
        }
        slot_filler_path = self.tmp_file_path / "slot_filler.json"
        self.assertJsonContent(slot_filler_path, expected_slot_filler_dict)

    def test_should_be_deserializable_before_fit(self):
        # Given
        features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        slot_filler_config = CRFSlotFillerConfig(
            feature_factory_configs=features_factories)
        slot_filler_dict = {
            "unit_name": "crf_slot_filler",
            "crf_model_file": None,
            "language_code": None,
            "intent": None,
            "slot_name_mapping": None,
            "config": slot_filler_config.to_dict()
        }
        metadata = {"unit_name": "crf_slot_filler"}
        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(self.tmp_file_path / "slot_filler.json",
                              slot_filler_dict)

        # When
        slot_filler = CRFSlotFiller.from_path(self.tmp_file_path)

        # Then
        expected_features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        expected_language = None
        expected_config = CRFSlotFillerConfig(
            feature_factory_configs=expected_features_factories)
        expected_intent = None
        expected_slot_name_mapping = None
        expected_crf_model = None

        self.assertEqual(slot_filler.crf_model, expected_crf_model)
        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping,
                         expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(),
                             slot_filler.config.to_dict())

    def test_should_be_serializable(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: my_intent
utterances:
- this is [slot1:entity1](my first entity)
- this is [slot2:entity2](second_entity)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(
            tagging_scheme=TaggingScheme.BILOU,
            feature_factory_configs=features_factories)
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(config, **shared)
        intent = "my_intent"
        slot_filler.fit(dataset, intent=intent)

        # When
        slot_filler.persist(self.tmp_file_path)

        # Then
        metadata_path = self.tmp_file_path / "metadata.json"
        self.assertJsonContent(metadata_path, {"unit_name": "crf_slot_filler"})

        expected_crf_file = Path(slot_filler.crf_model.modelfile.name).name
        self.assertTrue((self.tmp_file_path / expected_crf_file).exists())

        expected_feature_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1, "language_code": "en"},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        expected_config = CRFSlotFillerConfig(
            tagging_scheme=TaggingScheme.BILOU,
            feature_factory_configs=expected_feature_factories)
        expected_slot_filler_dict = {
            "crf_model_file": expected_crf_file,
            "language_code": "en",
            "config": expected_config.to_dict(),
            "intent": intent,
            "slot_name_mapping": {
                "slot1": "entity1",
                "slot2": "entity2",
            }
        }
        slot_filler_path = self.tmp_file_path / "slot_filler.json"
        self.assertJsonContent(slot_filler_path, expected_slot_filler_dict)

    def test_should_be_deserializable(self):
        # Given
        language = LANGUAGE_EN
        feature_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1, "language_code": language},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        slot_filler_config = CRFSlotFillerConfig(
            feature_factory_configs=feature_factories)
        slot_filler_dict = {
            "unit_name": "crf_slot_filler",
            "crf_model_file": "foobar.crfsuite",
            "language_code": "en",
            "intent": "dummy_intent_1",
            "slot_name_mapping": {
                "dummy_intent_1": {
                    "dummy_slot_name": "dummy_entity_1",
                }
            },
            "config": slot_filler_config.to_dict()
        }
        metadata = {"unit_name": "crf_slot_filler"}
        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(self.tmp_file_path / "slot_filler.json",
                              slot_filler_dict)
        self.writeFileContent(self.tmp_file_path / "foobar.crfsuite",
                              "foo bar")

        # When
        slot_filler = CRFSlotFiller.from_path(self.tmp_file_path)

        # Then
        expected_language = LANGUAGE_EN
        expected_feature_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1, "language_code": language},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        expected_config = CRFSlotFillerConfig(
            feature_factory_configs=expected_feature_factories)
        expected_intent = "dummy_intent_1"
        expected_slot_name_mapping = {
            "dummy_intent_1": {
                "dummy_slot_name": "dummy_entity_1",
            }
        }

        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping,
                         expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(),
                             slot_filler.config.to_dict())
        crf_path = Path(slot_filler.crf_model.modelfile.name)
        self.assertFileContent(crf_path, "foo bar")

    def test_should_be_serializable_when_fitted_without_slots(self):
        # Given
        features_factories = [
            {
                "factory_name": ShapeNgramFactory.name,
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": IsDigitFactory.name,
                "args": {},
                "offsets": [-1, 0]
            }
        ]
        config = CRFSlotFillerConfig(
            tagging_scheme=TaggingScheme.BILOU,
            feature_factory_configs=features_factories)
        dataset = {
            "language": "en",
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "This is an utterance without "
                                            "slots"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {}
        }

        slot_filler = CRFSlotFiller(config, **self.get_shared_data(dataset))
        slot_filler.fit(dataset, intent="intent1")

        # When
        slot_filler.persist(self.tmp_file_path)

        # Then
        metadata_path = self.tmp_file_path / "metadata.json"
        self.assertJsonContent(metadata_path, {"unit_name": "crf_slot_filler"})
        self.assertIsNone(slot_filler.crf_model)

    def test_should_be_deserializable_when_fitted_without_slots(self):
        # Given
        dataset = {
            "language": "en",
            "intents": {
                "intent1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "This is an utterance without "
                                            "slots"
                                }
                            ]
                        }
                    ]
                }
            },
            "entities": {}
        }

        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared)
        slot_filler.fit(dataset, intent="intent1")
        slot_filler.persist(self.tmp_file_path)
        loaded_slot_filler = CRFSlotFiller.from_path(
            self.tmp_file_path, **shared)

        # When
        slots = loaded_slot_filler.get_slots(
            "This is an utterance without slots")

        # Then
        self.assertListEqual([], slots)

    def test_should_be_serializable_into_bytearray(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me [number_of_cups:snips/number](one) cup of tea
- i want [number_of_cups] cups of tea please
- can you prepare [number_of_cups] cups of tea ?""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared).fit(dataset, "MakeTea")

        # When
        slot_filler_bytes = slot_filler.to_byte_array()
        loaded_slot_filler = CRFSlotFiller.from_byte_array(
            slot_filler_bytes, **shared)
        slots = loaded_slot_filler.get_slots("make me two cups of tea")

        # Then
        expected_slots = [
            unresolved_slot(match_range={START: 8, END: 11},
                            value='two',
                            entity='snips/number',
                            slot_name='number_of_cups')]
        self.assertListEqual(expected_slots, slots)

    def test_should_compute_features(self):
        # Given
        features_factories = [
            {
                "factory_name": NgramFactory.name,
                "args": {
                    "n": 1,
                    "use_stemming": False,
                    "common_words_gazetteer_name": None
                },
                "offsets": [0],
                "drop_out": 0.3
            },
        ]
        slot_filler_config = CRFSlotFillerConfig(
            feature_factory_configs=features_factories)

        tokens = tokenize("foo hello world bar", LANGUAGE_EN)
        dataset_stream = io.StringIO("""
---
type: intent
name: my_intent
utterances:
- this is [slot1:entity1](my first entity)
- this is [slot2:entity2](second_entity)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        shared = self.get_shared_data(
            dataset, CustomEntityParserUsage.WITHOUT_STEMS)
        slot_filler = CRFSlotFiller(slot_filler_config, **shared)
        slot_filler.fit(dataset, intent="my_intent")

        # When
        features_with_drop_out = slot_filler.compute_features(tokens, True)

        # Then
        expected_features = [
            {},
            {"ngram_1": "hello"},
            {"ngram_1": "world"},
            {"ngram_1": "bar"}
        ]

        self.assertListEqual(expected_features, features_with_drop_out)

    def test_should_fit_and_parse_empty_intent(self):
        # Given
        dataset = {
            "intents": {
                "dummy_intent": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": " "
                                }
                            ]
                        }
                    ]
                }
            },
            "language": "en",
            "entities": dict()
        }

        slot_filler = CRFSlotFiller(**self.get_shared_data(dataset))

        # When
        slot_filler.fit(dataset, "dummy_intent")
        slot_filler.get_slots("ya")

    def test___ensure_safe(self):
        unsafe_examples = [
            ([[]], [[]]),
            ([[], []], [[], []]),
        ]

        # We don't assert anything here but it segfault otherwise
        for x, y in unsafe_examples:
            x, y = _ensure_safe(x, y)
            model = CRF().fit(x, y)
            model.predict_single([""])
