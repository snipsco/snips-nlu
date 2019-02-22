# coding=utf-8
from __future__ import unicode_literals

import io

from mock import patch

from snips_nlu.constants import (
    DATA,
    ENTITY,
    LANGUAGE_EN,
    RES_ENTITY,
    RES_INTENT,
    RES_INTENT_NAME,
    RES_PROBA,
    RES_SLOTS,
    RES_VALUE,
    SLOT_NAME,
    TEXT,
)
from snips_nlu.dataset import Dataset
from snips_nlu.exceptions import IntentNotFoundError, NotTrained
from snips_nlu.intent_parser import (
    LookupIntentParser as DeterministicIntentParser
)
from snips_nlu.intent_parser.deterministic_intent_parser import (
    _deduplicate_overlapping_slots
)
from snips_nlu.pipeline.configs import (
    LookupIntentParserConfig as DeterministicIntentParserConfig
)
from snips_nlu.result import (
    empty_result,
    extraction_result,
    intent_classification_result,
    unresolved_slot,
)
from snips_nlu.tests.utils import FixtureTest, TEST_PATH


class TestDeterministicIntentParser(FixtureTest):
    def setUp(self):
        super(TestDeterministicIntentParser, self).setUp()
        slots_dataset_stream = io.StringIO(
            """
---
type: intent
name: dummy_intent_1
slots:
  - name: dummy_slot_name
    entity: dummy_entity_1
  - name: dummy_slot_name2
    entity: dummy_entity_2
  - name: startTime
    entity: snips/datetime
utterances:
  - >
      This is a [dummy_slot_name](dummy_1) query with another
      [dummy_slot_name2](dummy_2) [startTime](at 10p.m.) or
      [startTime](tomorrow)
  - "This    is  a  [dummy_slot_name](dummy_1) "
  - "[startTime](tomorrow evening) there is a [dummy_slot_name](dummy_1)"

---
type: entity
name: dummy_entity_1
automatically_extensible: no
values:
- [dummy_a, dummy 2a, dummy a, 2 dummy a]
- [dummy_b, dummy b, dummy_bb, dummy_b]
- dummy d

---
type: entity
name: dummy_entity_2
automatically_extensible: no
values:
- [dummy_c, 3p.m., dummy_cc, dummy c]"""
        )
        self.slots_dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json

    def test_should_parse_intent(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: intent1
utterances:
  - foo bar baz

---
type: intent
name: intent2
utterances:
  - foo bar ban"""
        )
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)
        text = "foo bar ban"

        # When
        parsing = parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="intent2", probability=probability
        )

        self.assertEqual(expected_intent, parsing[RES_INTENT])

    def test_should_parse_intent_with_filter(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: intent1
utterances:
  - foo bar baz

---
type: intent
name: intent2
utterances:
  - foo bar ban"""
        )
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)
        text = "foo bar ban"

        # When
        parsing = parser.parse(text, intents=["intent1"])

        # Then
        self.assertEqual(empty_result(text, 1.0), parsing)

    def test_should_parse_top_intents(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: intent1
utterances:
  - hello world

---
type: intent
name: intent2
utterances:
  - foo bar"""
        )
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)
        text = "hello world"

        # When
        results = parser.parse(text, top_n=3)

        # Then
        expected_intent = intent_classification_result(
            intent_name="intent1", probability=1.0
        )
        expected_results = [extraction_result(expected_intent, [])]
        self.assertEqual(expected_results, results)

    @patch("snips_nlu.intent_parser.lookup_intent_parser" ".get_stop_words")
    def test_should_parse_intent_with_stop_words(self, mock_get_stop_words):
        # Given
        mock_get_stop_words.return_value = {"a", "hey"}
        dataset = self.slots_dataset
        config = DeterministicIntentParserConfig(ignore_stop_words=True)
        parser = DeterministicIntentParser(config).fit(dataset)
        text = "Hey this is dummy_a query with another dummy_c at 10p.m. or " "at 12p.m."

        # When
        parsing = parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability
        )

        self.assertEqual(expected_intent, parsing[RES_INTENT])

    def test_should_ignore_ambiguous_utterances(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: dummy_intent_1
utterances:
  - Hello world

---
type: intent
name: dummy_intent_2
utterances:
  - Hello world"""
        )
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)
        text = "Hello world"

        # When
        res = parser.parse(text)

        # Then
        self.assertEqual(empty_result(text, 1.0), res)

    def test_should_not_parse_when_not_fitted(self):
        # Given
        parser = DeterministicIntentParser()

        # When / Then
        self.assertFalse(parser.fitted)
        with self.assertRaises(NotTrained):
            parser.parse("foobar")

    def test_should_parse_intent_after_deserialization(self):
        # Given
        dataset = self.slots_dataset
        shared = self.get_shared_data(dataset)
        parser = DeterministicIntentParser(**shared).fit(dataset)
        parser.persist(self.tmp_file_path)
        deserialized_parser = DeterministicIntentParser.from_path(
            self.tmp_file_path, **shared
        )
        text = "this is a dummy_a query with another dummy_c at 10p.m. or " "at 12p.m."

        # When
        parsing = deserialized_parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability
        )
        self.assertEqual(expected_intent, parsing[RES_INTENT])

    def test_should_parse_slots(self):
        # Given
        dataset = self.slots_dataset
        parser = DeterministicIntentParser().fit(dataset)
        texts = [
            (
                "this is a dummy a query with another dummy_c at 10p.m. or at"
                " 12p.m.",
                [
                    unresolved_slot(
                        match_range=(10, 17),
                        value="dummy a",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    ),
                    unresolved_slot(
                        match_range=(37, 44),
                        value="dummy_c",
                        entity="dummy_entity_2",
                        slot_name="dummy_slot_name2",
                    ),
                    unresolved_slot(
                        match_range=(45, 54),
                        value="at 10p.m.",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                    unresolved_slot(
                        match_range=(58, 67),
                        value="at 12p.m.",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                ],
            ),
            (
                "this, is,, a, dummy a query with another dummy_c at 10pm or "
                "at 12p.m.",
                [
                    unresolved_slot(
                        match_range=(14, 21),
                        value="dummy a",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    ),
                    unresolved_slot(
                        match_range=(41, 48),
                        value="dummy_c",
                        entity="dummy_entity_2",
                        slot_name="dummy_slot_name2",
                    ),
                    unresolved_slot(
                        match_range=(49, 56),
                        value="at 10pm",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                    unresolved_slot(
                        match_range=(60, 69),
                        value="at 12p.m.",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                ],
            ),
            (
                "this is a dummy b",
                [
                    unresolved_slot(
                        match_range=(10, 17),
                        value="dummy b",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    )
                ],
            ),
            (
                " this is a dummy b ",
                [
                    unresolved_slot(
                        match_range=(11, 18),
                        value="dummy b",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    )
                ],
            ),
            (
                " at 8am ’ there is a dummy  a",
                [
                    unresolved_slot(
                        match_range=(1, 7),
                        value="at 8am",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                    unresolved_slot(
                        match_range=(21, 29),
                        value="dummy  a",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    ),
                ],
            ),
        ]

        for text, expected_slots in texts:
            # When
            parsing = parser.parse(text)

            # Then
            self.assertListEqual(expected_slots, parsing[RES_SLOTS])

    def test_should_get_intents(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: greeting1
utterances:
  - Hello [name](John)

---
type: intent
name: greeting2
utterances:
  - How are you [name](Thomas)

---
type: intent
name: greeting3
utterances:
  - Hi [name](Robert)"""
        )

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)

        # When
        top_intents = parser.get_intents("Hello John")

        # Then
        expected_intents = [
            {RES_INTENT_NAME: "greeting1", RES_PROBA: 1.0},
            {RES_INTENT_NAME: "greeting2", RES_PROBA: 0.0},
            {RES_INTENT_NAME: "greeting3", RES_PROBA: 0.0},
            {RES_INTENT_NAME: None, RES_PROBA: 0.0},
        ]

        def sorting_key(intent_res):
            if intent_res[RES_INTENT_NAME] is None:
                return "null"
            return intent_res[RES_INTENT_NAME]

        sorted_expected_intents = sorted(expected_intents, key=sorting_key)
        sorted_intents = sorted(top_intents, key=sorting_key)
        self.assertEqual(expected_intents[0], top_intents[0])
        self.assertListEqual(sorted_expected_intents, sorted_intents)

    def test_should_get_slots(self):
        # Given
        slots_dataset_stream = io.StringIO(
            """
---
type: intent
name: greeting1
utterances:
  - Hello [name1](John)

---
type: intent
name: greeting2
utterances:
  - Hello [name2](Thomas)

---
type: intent
name: goodbye
utterances:
  - Goodbye [name](Eric)"""
        )
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)

        # When
        slots_greeting1 = parser.get_slots("Hello John", "greeting1")
        slots_greeting2 = parser.get_slots("Hello Thomas", "greeting2")
        slots_goodbye = parser.get_slots("Goodbye Eric", "greeting1")

        # Then
        self.assertEqual(1, len(slots_greeting1))
        self.assertEqual(1, len(slots_greeting2))
        self.assertEqual(0, len(slots_goodbye))

        self.assertEqual("John", slots_greeting1[0][RES_VALUE])
        self.assertEqual("name1", slots_greeting1[0][RES_ENTITY])
        self.assertEqual("Thomas", slots_greeting2[0][RES_VALUE])
        self.assertEqual("name2", slots_greeting2[0][RES_ENTITY])

    def test_should_get_no_slots_with_none_intent(self):
        # Given
        slots_dataset_stream = io.StringIO(
            """
---
type: intent
name: greeting
utterances:
  - Hello [name](John)"""
        )
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)

        # When
        slots = parser.get_slots("Hello John", None)

        # Then
        self.assertListEqual([], slots)

    def test_get_slots_should_raise_with_unknown_intent(self):
        # Given
        slots_dataset_stream = io.StringIO(
            """
---
type: intent
name: greeting1
utterances:
  - Hello [name1](John)

---
type: intent
name: goodbye
utterances:
  - Goodbye [name](Eric)"""
        )
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = DeterministicIntentParser().fit(dataset)

        # When / Then
        with self.assertRaises(IntentNotFoundError):
            parser.get_slots("Hello John", "greeting3")

    def test_should_parse_slots_after_deserialization(self):
        # Given
        dataset = self.slots_dataset
        shared = self.get_shared_data(dataset)
        parser = DeterministicIntentParser(**shared).fit(dataset)
        parser.persist(self.tmp_file_path)
        deserialized_parser = DeterministicIntentParser.from_path(
            self.tmp_file_path, **shared
        )

        texts = [
            (
                "this is a dummy a query with another dummy_c at 10p.m. or at"
                " 12p.m.",
                [
                    unresolved_slot(
                        match_range=(10, 17),
                        value="dummy a",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    ),
                    unresolved_slot(
                        match_range=(37, 44),
                        value="dummy_c",
                        entity="dummy_entity_2",
                        slot_name="dummy_slot_name2",
                    ),
                    unresolved_slot(
                        match_range=(45, 54),
                        value="at 10p.m.",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                    unresolved_slot(
                        match_range=(58, 67),
                        value="at 12p.m.",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                ],
            ),
            (
                "this, is,, a, dummy a query with another dummy_c at 10pm or "
                "at 12p.m.",
                [
                    unresolved_slot(
                        match_range=(14, 21),
                        value="dummy a",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    ),
                    unresolved_slot(
                        match_range=(41, 48),
                        value="dummy_c",
                        entity="dummy_entity_2",
                        slot_name="dummy_slot_name2",
                    ),
                    unresolved_slot(
                        match_range=(49, 56),
                        value="at 10pm",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                    unresolved_slot(
                        match_range=(60, 69),
                        value="at 12p.m.",
                        entity="snips/datetime",
                        slot_name="startTime",
                    ),
                ],
            ),
            (
                "this is a dummy b",
                [
                    unresolved_slot(
                        match_range=(10, 17),
                        value="dummy b",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    )
                ],
            ),
            (
                " this is a dummy b ",
                [
                    unresolved_slot(
                        match_range=(11, 18),
                        value="dummy b",
                        entity="dummy_entity_1",
                        slot_name="dummy_slot_name",
                    )
                ],
            ),
        ]

        for text, expected_slots in texts:
            # When
            parsing = deserialized_parser.parse(text)

            # Then
            self.assertListEqual(expected_slots, parsing[RES_SLOTS])

    def test_should_be_serializable_into_bytearray(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: MakeTea
utterances:
- make me [number_of_cups:snips/number](one) cup of tea
- i want [number_of_cups] cups of tea please
- can you prepare [number_of_cups] cup of tea ?

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](two) cups of coffee
- brew [number_of_cups] cups of coffee
- can you prepare [number_of_cups] cup of coffee"""
        )
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        intent_parser = DeterministicIntentParser(**shared).fit(dataset)

        # When
        intent_parser_bytes = intent_parser.to_byte_array()
        loaded_intent_parser = DeterministicIntentParser.from_byte_array(
            intent_parser_bytes, **shared
        )
        result = loaded_intent_parser.parse("make me two cups of coffee")

        # Then
        self.assertEqual("MakeCoffee", result[RES_INTENT][RES_INTENT_NAME])

    def test_should_parse_naughty_strings(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: my_intent
utterances:
- this is [slot1:entity1](my first entity)
- this is [slot2:entity2](second_entity)"""
        )
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        naughty_strings_path = TEST_PATH / "resources" / "naughty_strings.txt"
        with naughty_strings_path.open(encoding="utf8") as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        # When
        parser = DeterministicIntentParser().fit(dataset)

        # Then
        for s in naughty_strings:
            with self.fail_if_exception("Exception raised"):
                parser.parse(s)

    def test_should_fit_with_naughty_strings_no_tags(self):
        # Given
        naughty_strings_path = TEST_PATH / "resources" / "naughty_strings.txt"
        with naughty_strings_path.open(encoding="utf8") as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        utterances = [
            {DATA: [{TEXT: naughty_string}]} for naughty_string in naughty_strings
        ]

        # When
        naughty_dataset = {
            "intents": {"naughty_intent": {"utterances": utterances}},
            "entities": dict(),
            "language": "en",
        }

        # Then
        with self.fail_if_exception("Exception raised"):
            DeterministicIntentParser().fit(naughty_dataset)

    def test_should_fit_and_parse_with_non_ascii_tags(self):
        # Given
        inputs = ["string%s" % i for i in range(10)]
        utterances = [
            {
                DATA: [
                    {
                        TEXT: string,
                        ENTITY: "non_ascìi_entïty",
                        SLOT_NAME: "non_ascìi_slöt",
                    }
                ]
            }
            for string in inputs
        ]

        # When
        naughty_dataset = {
            "intents": {"naughty_intent": {"utterances": utterances}},
            "entities": {
                "non_ascìi_entïty": {
                    "use_synonyms": False,
                    "automatically_extensible": True,
                    "matching_strictness": 1.0,
                    "data": [],
                }
            },
            "language": "en",
        }

        # Then
        with self.fail_if_exception("Exception raised"):
            parser = DeterministicIntentParser().fit(naughty_dataset)
            parsing = parser.parse("string0")

            expected_slot = {
                "entity": "non_ascìi_entïty",
                "range": {"start": 0, "end": 7},
                "slotName": "non_ascìi_slöt",
                "value": "string0",
            }
            intent_name = parsing[RES_INTENT][RES_INTENT_NAME]
            self.assertEqual("naughty_intent", intent_name)
            self.assertListEqual([expected_slot], parsing[RES_SLOTS])

    def test_should_be_serializable_before_fitting(self):
        # Given
        config = DeterministicIntentParserConfig(ignore_stop_words=True)
        parser = DeterministicIntentParser(config=config)

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_dict = {
            "config": {
                "unit_name": "lookup_intent_parser",
                "ignore_stop_words": True,
            },
            "language_code": None,
            "intents_names": [],
            "map": None,
            "slots_names": [],
        }

        metadata = {"unit_name": "lookup_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json", expected_dict)

    @patch("snips_nlu.intent_parser.lookup_intent_parser.get_stop_words")
    def test_should_be_serializable(self, mock_get_stop_words):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: searchFlight
slots:
  - name: origin
    entity: city
  - name: destination
    entity: city
utterances:
  - find me a flight from [origin](Paris) to [destination](New York)
  - I need a flight to [destination](Berlin)

---
type: entity
name: city
values:
  - london
  - [new york, big apple]
  - [paris, city of lights]
            """
        )

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        mock_get_stop_words.return_value = {"a", "me"}
        config = DeterministicIntentParserConfig(ignore_stop_words=True)
        parser = DeterministicIntentParser(config=config).fit(dataset)

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_dict = {
            "config": {
                "unit_name": "lookup_intent_parser",
                "ignore_stop_words": True,
            },
            "intents_names": ["searchFlight"],
            "language_code": "en",
            "map": {
                "findflightfrom%city%to%city%": [0, [0, 1]],
                "ineedflightto%city%": [0, [1]],
            },
            "slots_names": ["origin", "destination"],
        }
        metadata = {"unit_name": "lookup_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json", expected_dict)

    def test_should_be_deserializable(self):
        # Given
        parser_dict = {
            "config": {},
            "language_code": "en",
            "map": None,
            "slots_names": [],
            "intents_names": [],
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "deterministic_intent_parser"}
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json", parser_dict)
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)

        # When
        parser = DeterministicIntentParser.from_path(self.tmp_file_path)

        # Then
        patterns = {"my_intent": ["(?P<hello_group>hello?)", "(?P<world_group>world$)"]}
        group_names_to_slot_names = {
            "hello_group": "hello_slot",
            "world_group": "world_slot",
        }
        slot_names_to_entities = {
            "my_intent": {"hello_slot": "hello_entity", "world_slot": "world_entity"}
        }
        config = DeterministicIntentParserConfig()
        expected_parser = DeterministicIntentParser(config=config)
        expected_parser.language = LANGUAGE_EN
        expected_parser.group_names_to_slot_names = group_names_to_slot_names
        expected_parser.slot_names_to_entities = slot_names_to_entities
        expected_parser.patterns = patterns

        self.assertEqual(parser.to_dict(), expected_parser.to_dict())

    def test_should_be_deserializable_before_fitting(self):
        # Given
        parser_dict = {
            "config": {},
            "language_code": None,
            "map": None,
            "slots_names": [],
            "intents_names": [],
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "dict_deterministic_intent_parser"}
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json", parser_dict)
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)

        # When
        parser = DeterministicIntentParser.from_path(self.tmp_file_path)

        # Then
        config = DeterministicIntentParserConfig()
        expected_parser = DeterministicIntentParser(config=config)
        self.assertEqual(parser.to_dict(), expected_parser.to_dict())

    def test_should_deduplicate_overlapping_slots(self):
        # Given
        language = LANGUAGE_EN
        slots = [
            unresolved_slot([0, 3], "kid", "e", "s1"),
            unresolved_slot([4, 8], "loco", "e1", "s2"),
            unresolved_slot([0, 8], "kid loco", "e1", "s3"),
            unresolved_slot([9, 13], "song", "e2", "s4"),
        ]

        # When
        deduplicated_slots = _deduplicate_overlapping_slots(slots, language)

        # Then
        expected_slots = [
            unresolved_slot([0, 8], "kid loco", "e1", "s3"),
            unresolved_slot([9, 13], "song", "e2", "s4"),
        ]
        self.assertSequenceEqual(deduplicated_slots, expected_slots)
