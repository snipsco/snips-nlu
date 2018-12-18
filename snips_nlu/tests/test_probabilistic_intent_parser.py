from __future__ import unicode_literals

import io

from future.utils import itervalues
from mock import patch

from snips_nlu.constants import (
    RES_ENTITY, RES_INTENT, RES_INTENT_NAME, RES_SLOTS, RES_VALUE)
from snips_nlu.dataset import Dataset
from snips_nlu.exceptions import IntentNotFoundError, NotTrained
from snips_nlu.intent_classifier import (
    IntentClassifier, LogRegIntentClassifier)
from snips_nlu.intent_parser import ProbabilisticIntentParser
from snips_nlu.pipeline.configs import (
    CRFSlotFillerConfig, LogRegIntentClassifierConfig,
    ProbabilisticIntentParserConfig)
from snips_nlu.result import unresolved_slot
from snips_nlu.slot_filler import CRFSlotFiller, SlotFiller
from snips_nlu.tests.utils import (
    FixtureTest, MockIntentClassifier, MockSlotFiller)


class TestProbabilisticIntentParser(FixtureTest):
    def setUp(self):
        super(TestProbabilisticIntentParser, self).setUp()

    def test_should_parse(self):
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - "[slot1:entity1](foo) bar"

---
type: intent
name: intent2
utterances:
  - foo bar [slot2:entity2](baz)

---
type: intent
name: intent3
utterances:
  - foz for [slot3:entity3](baz)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        classifier_config = LogRegIntentClassifierConfig(random_seed=42)
        slot_filler_config = CRFSlotFillerConfig(random_seed=42)
        parser_config = ProbabilisticIntentParserConfig(
            classifier_config, slot_filler_config)
        parser = ProbabilisticIntentParser(parser_config).fit(dataset)
        text = "foo bar baz"

        # When
        result = parser.parse(text)

        # Then
        expected_slots = [
            unresolved_slot((8, 11), "baz", "entity2", "slot2")
        ]

        self.assertEqual("intent2", result[RES_INTENT][RES_INTENT_NAME])
        self.assertEqual(expected_slots, result[RES_SLOTS])

    def test_should_parse_with_filter(self):
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - "[slot1:entity1](foo) bar"

---
type: intent
name: intent2
utterances:
  - foo bar [slot2:entity2](baz)

---
type: intent
name: intent3
utterances:
  - foz for [slot3:entity3](baz)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        classifier_config = LogRegIntentClassifierConfig(random_seed=42)
        slot_filler_config = CRFSlotFillerConfig(random_seed=42)
        parser_config = ProbabilisticIntentParserConfig(
            classifier_config, slot_filler_config)
        parser = ProbabilisticIntentParser(parser_config).fit(dataset)
        text = "foo bar baz"

        # When
        result = parser.parse(text, intents=["intent1", "intent3"])

        # Then
        expected_slots = [
            unresolved_slot((0, 3), "foo", "entity1", "slot1")
        ]

        self.assertEqual("intent1", result[RES_INTENT][RES_INTENT_NAME])
        self.assertEqual(expected_slots, result[RES_SLOTS])

    def test_should_parse_top_intents(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - "[entity1](foo) bar"

---
type: intent
name: intent2
utterances:
  - foo bar [entity2](baz)

---
type: intent
name: intent3
utterances:
  - foz for [entity3](baz)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        classifier_config = LogRegIntentClassifierConfig(random_seed=42)
        slot_filler_config = CRFSlotFillerConfig(random_seed=42)
        parser_config = ProbabilisticIntentParserConfig(
            classifier_config, slot_filler_config)
        parser = ProbabilisticIntentParser(parser_config).fit(dataset)
        text = "foo bar baz"

        # When
        results = parser.parse(text, top_n=2)
        intents = [res[RES_INTENT][RES_INTENT_NAME] for res in results]
        entities = [[s[RES_VALUE] for s in res[RES_SLOTS]] for res in results]

        # Then
        expected_intents = ["intent2", "intent1"]
        expected_entities = [["baz"], ["foo"]]

        self.assertListEqual(expected_intents, intents)
        self.assertListEqual(expected_entities, entities)

    def test_should_get_intents(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - yala yili

---
type: intent
name: intent2
utterances:
  - yala yili yulu

---
type: intent
name: intent3
utterances:
  - yili yulu yele""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        classifier_config = LogRegIntentClassifierConfig(random_seed=42)
        parser_config = ProbabilisticIntentParserConfig(classifier_config)
        parser = ProbabilisticIntentParser(parser_config).fit(dataset)
        text = "yala yili yulu"

        # When
        results = parser.get_intents(text)
        intents = [res[RES_INTENT_NAME] for res in results]

        # Then
        expected_intents = ["intent2", "intent1", "intent3", None]

        self.assertEqual(expected_intents, intents)

    def test_should_get_slots(self):
        # Given
        slots_dataset_stream = io.StringIO("""
---
type: intent
name: greeting1
utterances:
  - Hello [name1](John)

---
type: intent
name: greeting2
utterances:
  - Hello [name2](John)

---
type: intent
name: greeting3
utterances:
  - Hello John""")
        dataset = Dataset.from_yaml_files("en",
                                          [slots_dataset_stream]).json
        parser = ProbabilisticIntentParser().fit(dataset)

        # When
        slots_greeting1 = parser.get_slots("Hello John", "greeting1")
        slots_greeting2 = parser.get_slots("Hello John", "greeting2")
        slots_goodbye = parser.get_slots("Hello John", "greeting3")

        # Then
        self.assertEqual(1, len(slots_greeting1))
        self.assertEqual(1, len(slots_greeting2))
        self.assertEqual(0, len(slots_goodbye))

        self.assertEqual("John", slots_greeting1[0][RES_VALUE])
        self.assertEqual("name1", slots_greeting1[0][RES_ENTITY])
        self.assertEqual("John", slots_greeting2[0][RES_VALUE])
        self.assertEqual("name2", slots_greeting2[0][RES_ENTITY])

    def test_should_get_no_slots_with_none_intent(self):
        # Given
        slots_dataset_stream = io.StringIO("""
---
type: intent
name: greeting
utterances:
  - Hello [name](John)""")
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = ProbabilisticIntentParser().fit(dataset)

        # When
        slots = parser.get_slots("Hello John", None)

        # Then
        self.assertListEqual([], slots)

    def test_get_slots_should_raise_with_unknown_intent(self):
        # Given
        slots_dataset_stream = io.StringIO("""
---
type: intent
name: greeting1
utterances:
  - Hello [name1](John)

---
type: intent
name: goodbye
utterances:
  - Goodbye [name](Eric)""")
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = ProbabilisticIntentParser().fit(dataset)

        # When / Then
        with self.assertRaises(IntentNotFoundError):
            parser.get_slots("Hello John", "greeting3")

    def test_should_retrain_intent_classifier_when_force_retrain(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = ProbabilisticIntentParser()
        intent_classifier = LogRegIntentClassifier()
        intent_classifier.fit(dataset)
        parser.intent_classifier = intent_classifier

        # When / Then
        with patch("snips_nlu.intent_classifier.log_reg_classifier"
                   ".LogRegIntentClassifier.fit") as mock_fit:
            parser.fit(dataset, force_retrain=True)
            mock_fit.assert_called_once()

    def test_should_not_retrain_intent_classifier_when_no_force_retrain(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = ProbabilisticIntentParser()
        intent_classifier = LogRegIntentClassifier()
        intent_classifier.fit(dataset)
        parser.intent_classifier = intent_classifier

        # When / Then
        with patch("snips_nlu.intent_classifier.log_reg_classifier"
                   ".LogRegIntentClassifier.fit") as mock_fit:
            parser.fit(dataset, force_retrain=False)
            mock_fit.assert_not_called()

    def test_should_retrain_slot_filler_when_force_retrain(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = ProbabilisticIntentParser()
        slot_filler = CRFSlotFiller()
        slot_filler.fit(dataset, "MakeCoffee")
        parser.slot_fillers["MakeCoffee"] = slot_filler

        # When / Then
        with patch("snips_nlu.slot_filler.crf_slot_filler.CRFSlotFiller.fit") \
                as mock_fit:
            parser.fit(dataset, force_retrain=True)
            self.assertEqual(2, mock_fit.call_count)

    def test_should_not_retrain_slot_filler_when_no_force_retrain(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = ProbabilisticIntentParser()
        slot_filler = CRFSlotFiller()
        slot_filler.fit(dataset, "MakeCoffee")
        parser.slot_fillers["MakeCoffee"] = slot_filler

        # When / Then
        with patch("snips_nlu.slot_filler.crf_slot_filler.CRFSlotFiller.fit") \
                as mock_fit:
            parser.fit(dataset, force_retrain=False)
            self.assertEqual(1, mock_fit.call_count)

    def test_should_not_parse_when_not_fitted(self):
        # Given
        parser = ProbabilisticIntentParser()

        # When / Then
        self.assertFalse(parser.fitted)
        with self.assertRaises(NotTrained):
            parser.parse("foobar")

    def test_should_be_serializable_before_fitting(self):
        # Given
        parser = ProbabilisticIntentParser()

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_parser_dict = {
            "config": {
                "unit_name": "probabilistic_intent_parser",
                "slot_filler_config": CRFSlotFillerConfig().to_dict(),
                "intent_classifier_config":
                    LogRegIntentClassifierConfig().to_dict()
            },
            "slot_fillers": []
        }
        metadata = {"unit_name": "probabilistic_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json",
                               expected_parser_dict)

    def test_should_be_deserializable_before_fitting(self):
        # When
        config = ProbabilisticIntentParserConfig().to_dict()
        parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "config": config,
            "intent_classifier": None,
            "slot_fillers": dict(),
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "probabilistic_intent_parser"}
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json",
                              parser_dict)

        # When
        parser = ProbabilisticIntentParser.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(parser.config.to_dict(), config)
        self.assertIsNone(parser.intent_classifier)
        self.assertDictEqual(dict(), parser.slot_fillers)

    def test_should_be_serializable(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        @IntentClassifier.register("my_intent_classifier", True)
        class MyIntentClassifier(MockIntentClassifier):
            pass

        @SlotFiller.register("my_slot_filler", True)
        class MySlotFiller(MockSlotFiller):
            pass

        parser_config = ProbabilisticIntentParserConfig(
            intent_classifier_config="my_intent_classifier",
            slot_filler_config="my_slot_filler"
        )
        parser = ProbabilisticIntentParser(parser_config).fit(dataset)

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_parser_config = {
            "unit_name": "probabilistic_intent_parser",
            "slot_filler_config": {"unit_name": "my_slot_filler"},
            "intent_classifier_config": {"unit_name": "my_intent_classifier"}
        }
        expected_parser_dict = {
            "config": expected_parser_config,
            "slot_fillers": [
                {
                    "intent": "MakeCoffee",
                    "slot_filler_name": "slot_filler_0"
                },
                {
                    "intent": "MakeTea",
                    "slot_filler_name": "slot_filler_1"
                }
            ]
        }
        metadata = {"unit_name": "probabilistic_intent_parser"}
        metadata_slot_filler = {"unit_name": "my_slot_filler"}
        metadata_intent_classifier = {"unit_name": "my_intent_classifier"}

        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json",
                               expected_parser_dict)
        self.assertJsonContent(
            self.tmp_file_path / "intent_classifier" / "metadata.json",
            metadata_intent_classifier)
        self.assertJsonContent(
            self.tmp_file_path / "slot_filler_0" / "metadata.json",
            metadata_slot_filler)
        self.assertJsonContent(
            self.tmp_file_path / "slot_filler_1" / "metadata.json",
            metadata_slot_filler)

    def test_should_be_deserializable(self):
        # When
        @IntentClassifier.register("my_intent_classifier", True)
        class MyIntentClassifier(MockIntentClassifier):
            pass

        @SlotFiller.register("my_slot_filler", True)
        class MySlotFiller(MockSlotFiller):
            pass

        parser_config = {
            "unit_name": "probabilistic_intent_parser",
            "intent_classifier_config": {
                "unit_name": "my_intent_classifier"
            },
            "slot_filler_config": {
                "unit_name": "my_slot_filler"
            }
        }
        parser_dict = {
            "unit_name": "probabilistic_intent_parser",
            "slot_fillers": [
                {
                    "intent": "MakeCoffee",
                    "slot_filler_name": "slot_filler_MakeCoffee"
                },
                {
                    "intent": "MakeTea",
                    "slot_filler_name": "slot_filler_MakeTea"
                }
            ],
            "config": parser_config,
        }
        self.tmp_file_path.mkdir()
        (self.tmp_file_path / "intent_classifier").mkdir()
        (self.tmp_file_path / "slot_filler_MakeCoffee").mkdir()
        (self.tmp_file_path / "slot_filler_MakeTea").mkdir()
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json",
                              parser_dict)
        self.writeJsonContent(
            self.tmp_file_path / "intent_classifier" / "metadata.json",
            {"unit_name": "my_intent_classifier"})
        self.writeJsonContent(
            self.tmp_file_path / "slot_filler_MakeCoffee" / "metadata.json",
            {"unit_name": "my_slot_filler"})
        self.writeJsonContent(
            self.tmp_file_path / "slot_filler_MakeTea" / "metadata.json",
            {"unit_name": "my_slot_filler"})

        # When
        parser = ProbabilisticIntentParser.from_path(self.tmp_file_path)

        # Then
        self.assertDictEqual(parser.config.to_dict(), parser_config)
        self.assertIsInstance(parser.intent_classifier, MyIntentClassifier)
        self.assertListEqual(sorted(parser.slot_fillers),
                             ["MakeCoffee", "MakeTea"])
        for slot_filler in itervalues(parser.slot_fillers):
            self.assertIsInstance(slot_filler, MySlotFiller)

    def test_should_be_serializable_into_bytearray(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        intent_parser = ProbabilisticIntentParser().fit(dataset)
        builtin_entity_parser = intent_parser.builtin_entity_parser
        custom_entity_parser = intent_parser.custom_entity_parser

        # When
        intent_parser_bytes = intent_parser.to_byte_array()
        loaded_intent_parser = ProbabilisticIntentParser.from_byte_array(
            intent_parser_bytes,
            builtin_entity_parser=builtin_entity_parser,
            custom_entity_parser=custom_entity_parser
        )
        result = loaded_intent_parser.parse("make me two cups of tea")

        # Then
        self.assertEqual("MakeTea", result[RES_INTENT][RES_INTENT_NAME])

    def test_fitting_should_be_reproducible_after_serialization(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        seed1 = 666
        seed2 = 42
        config = ProbabilisticIntentParserConfig(
            intent_classifier_config=LogRegIntentClassifierConfig(
                random_seed=seed1),
            slot_filler_config=CRFSlotFillerConfig(random_seed=seed2)
        )
        parser = ProbabilisticIntentParser(config)
        parser.persist(self.tmp_file_path)

        # When
        fitted_parser_1 = ProbabilisticIntentParser.from_path(
            self.tmp_file_path).fit(dataset)

        fitted_parser_2 = ProbabilisticIntentParser.from_path(
            self.tmp_file_path).fit(dataset)

        # Then
        feature_weights_1 = fitted_parser_1.slot_fillers[
            "MakeTea"].crf_model.state_features_
        feature_weights_2 = fitted_parser_2.slot_fillers[
            "MakeTea"].crf_model.state_features_
        self.assertEqual(feature_weights_1, feature_weights_2)
