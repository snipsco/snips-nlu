# coding=utf-8
from __future__ import unicode_literals

import io
from copy import deepcopy

from mock import patch
from snips_nlu_utils import hash_str

from snips_nlu.constants import (
    DATA, ENTITY, RES_ENTITY, RES_INTENT, RES_INTENT_NAME,
    RES_PROBA, RES_SLOTS, RES_VALUE, SLOT_NAME, TEXT, STOP_WORDS)
from snips_nlu.dataset import Dataset
from snips_nlu.entity_parser import BuiltinEntityParser
from snips_nlu.exceptions import IntentNotFoundError, NotTrained
from snips_nlu.intent_parser import LookupIntentParser
from snips_nlu.intent_parser.lookup_intent_parser import _get_entity_scopes
from snips_nlu.pipeline.configs import LookupIntentParserConfig
from snips_nlu.result import (
    empty_result, extraction_result, intent_classification_result,
    unresolved_slot, parsing_result)
from snips_nlu.tests.utils import FixtureTest, TEST_PATH, EntityParserMock


class TestLookupIntentParser(FixtureTest):
    def setUp(self):
        super(TestLookupIntentParser, self).setUp()
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
- [dummy_c, 3p.m., dummy_cc, dummy c]""")
        self.slots_dataset = Dataset.from_yaml_files(
            "en", [slots_dataset_stream]).json

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
  - foo bar ban""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "foo bar ban"

        # When
        parsing = parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="intent2", probability=probability)

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
  - foo bar ban""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "foo bar ban"

        # When
        parsing = parser.parse(text, intents=["intent1"])

        # Then
        self.assertEqual(empty_result(text, 1.0), parsing)

    def test_should_parse_top_intents(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - meeting [time:snips/datetime](today)

---
type: intent
name: intent2
utterances:
  - meeting tomorrow

---
type: intent
name: intent3
utterances:
  - "[event_type](call) [time:snips/datetime](at 9pm)"

---
type: entity
name: event_type
values:
  - meeting
  - feedback session""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "meeting tomorrow"

        # When
        results = parser.parse(text, top_n=3)

        # Then
        time_slot = {
            "entity": "snips/datetime",
            "range": {"end": 16, "start": 8},
            "slotName": "time",
            "value": "tomorrow"
        }
        event_slot = {
            "entity": "event_type",
            "range": {"end": 7, "start": 0},
            "slotName": "event_type",
            "value": "meeting"
        }
        weight_intent_1 = 1. / 2.
        weight_intent_2 = 1.
        weight_intent_3 = 1. / 3.
        total_weight = weight_intent_1 + weight_intent_2 + weight_intent_3
        proba_intent2 = weight_intent_2 / total_weight
        proba_intent1 = weight_intent_1 / total_weight
        proba_intent3 = weight_intent_3 / total_weight
        expected_results = [
            extraction_result(
                intent_classification_result(
                    intent_name="intent2", probability=proba_intent2),
                slots=[]),
            extraction_result(
                intent_classification_result(
                    intent_name="intent1", probability=proba_intent1),
                slots=[time_slot]),
            extraction_result(
                intent_classification_result(
                    intent_name="intent3", probability=proba_intent3),
                slots=[event_slot, time_slot])
        ]
        self.assertEqual(expected_results, results)

    @patch("snips_nlu.intent_parser.lookup_intent_parser" ".get_stop_words")
    def test_should_parse_intent_with_stop_words(self, mock_get_stop_words):
        # Given
        mock_get_stop_words.return_value = {"a", "hey"}
        dataset = self.slots_dataset
        config = LookupIntentParserConfig(ignore_stop_words=True)
        parser = LookupIntentParser(config).fit(dataset)
        text = "Hey this is dummy_a query with another dummy_c at 10p.m. " \
               "or at 12p.m."

        # When
        parsing = parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability)

        self.assertEqual(expected_intent, parsing[RES_INTENT])

    def test_should_parse_intent_with_duplicated_slot_names(self):
        # Given
        slots_dataset_stream = io.StringIO("""
---
type: intent
name: math_operation
slots:
  - name: number
    entity: snips/number
utterances:
  - what is [number](one) plus [number](one)""")
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "what is one plus one"

        # When
        parsing = parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="math_operation", probability=probability)
        expected_slots = [
            {
                "entity": "snips/number",
                "range": {"end": 11, "start": 8},
                "slotName": "number",
                "value": "one"
            },
            {
                "entity": "snips/number",
                "range": {"end": 20, "start": 17},
                "slotName": "number",
                "value": "one"
            }
        ]

        self.assertDictEqual(expected_intent, parsing[RES_INTENT])
        self.assertListEqual(expected_slots, parsing[RES_SLOTS])

    def test_should_parse_intent_with_ambivalent_words(self):
        # Given
        slots_dataset_stream = io.StringIO("""
---
type: intent
name: give_flower
utterances:
  - give a rose to [name](emily)
  - give a daisy to [name](tom)
  - give a tulip to [name](daisy)
  """)
        dataset = Dataset.from_yaml_files("en",
                                          [slots_dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "give a daisy to emily"

        # When
        parsing = parser.parse(text)

        # Then
        expected_intent = intent_classification_result(
            intent_name="give_flower", probability=1.0)
        expected_slots = [
            {
                "entity": "name",
                "range": {"end": 21, "start": 16},
                "slotName": "name",
                "value": "emily"
            }
        ]

        self.assertDictEqual(expected_intent, parsing[RES_INTENT])
        self.assertListEqual(expected_slots, parsing[RES_SLOTS])

    def test_should_ignore_completely_ambiguous_utterances(self):
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
  - Hello world""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "Hello world"

        # When
        res = parser.parse(text)

        # Then
        self.assertEqual(empty_result(text, 1.0), res)

    def test_should_ignore_very_ambiguous_utterances(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent_1
utterances:
  - "[event_type](meeting) tomorrow"

---
type: intent
name: intent_2
utterances:
  - call [time:snips/datetime](today)

---
type: entity
name: event_type
values:
  - call
  - diner""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "call tomorrow"

        # When
        res = parser.parse(text)

        # Then
        self.assertEqual(empty_result(text, 1.0), res)

    def test_should_parse_slightly_ambiguous_utterances(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent_1
utterances:
  - call tomorrow

---
type: intent
name: intent_2
utterances:
  - call [time:snips/datetime](today)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)
        text = "call tomorrow"

        # When
        res = parser.parse(text)

        # Then
        expected_intent = intent_classification_result(
            intent_name="intent_1", probability=2. / 3.)
        expected_result = parsing_result(text, expected_intent, [])
        self.assertEqual(expected_result, res)

    def test_should_not_parse_when_not_fitted(self):
        # Given
        parser = LookupIntentParser()

        # When / Then
        self.assertFalse(parser.fitted)
        with self.assertRaises(NotTrained):
            parser.parse("foobar")

    def test_should_parse_intent_after_deserialization(self):
        # Given
        dataset = self.slots_dataset
        shared = self.get_shared_data(dataset)
        parser = LookupIntentParser(**shared).fit(dataset)
        parser.persist(self.tmp_file_path)
        deserialized_parser = LookupIntentParser.from_path(
            self.tmp_file_path, **shared)
        text = "this is a dummy_a query with another dummy_c at 10p.m. or " \
               "at 12p.m."

        # When
        parsing = deserialized_parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability)
        self.assertEqual(expected_intent, parsing[RES_INTENT])

    def test_should_parse_slots(self):
        # Given
        dataset = self.slots_dataset
        parser = LookupIntentParser().fit(dataset)
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

    def test_should_parse_stop_words_slots(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: search
utterances:
  - search
  - search [search_object](this)
  - search [search_object](a cat)

---
type: entity
name: search_object
values:
  - [this thing, that]
  """)

        resources = deepcopy(self.get_resources("en"))
        resources[STOP_WORDS] = {"a", "this", "that"}
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser_config = LookupIntentParserConfig(ignore_stop_words=True)
        parser = LookupIntentParser(config=parser_config, resources=resources)
        parser.fit(dataset)

        # When
        res_1 = parser.parse("search this")
        res_2 = parser.parse("search that")

        # Then
        expected_intent = intent_classification_result(
            intent_name="search", probability=1.0)
        expected_slots_1 = [
            unresolved_slot(match_range=(7, 11), value="this",
                            entity="search_object",
                            slot_name="search_object")
        ]
        expected_slots_2 = [
            unresolved_slot(match_range=(7, 11), value="that",
                            entity="search_object",
                            slot_name="search_object")
        ]
        self.assertEqual(expected_intent, res_1[RES_INTENT])
        self.assertEqual(expected_intent, res_2[RES_INTENT])
        self.assertListEqual(expected_slots_1, res_1[RES_SLOTS])
        self.assertListEqual(expected_slots_2, res_2[RES_SLOTS])

    def test_should_get_intents(self):
        # Given
        dataset_stream = io.StringIO(
            """
---
type: intent
name: greeting1
utterances:
  - Hello John

---
type: intent
name: greeting2
utterances:
  - Hello [name](John)

---
type: intent
name: greeting3
utterances:
  - "[greeting](Hello) [name](John)"
        """)

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)

        # When
        top_intents = parser.get_intents("Hello John")

        # Then
        expected_intents = [
            {
                RES_INTENT_NAME: "greeting1",
                RES_PROBA: 1. / (1. + 1 / 2 + 1 / 3)
            },
            {
                RES_INTENT_NAME: "greeting2",
                RES_PROBA: (1. / 2.) / (1. + 1 / 2 + 1 / 3)
            },
            {
                RES_INTENT_NAME: "greeting3",
                RES_PROBA: (1. / 3.) / (1. + 1 / 2 + 1 / 3)
            },
            {
                RES_INTENT_NAME: None,
                RES_PROBA: 0.0
            },
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
  - Goodbye [name](Eric)""")
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)

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
  - Hello [name](John)""")
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)

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
  - Goodbye [name](Eric)""")
        dataset = Dataset.from_yaml_files("en", [slots_dataset_stream]).json
        parser = LookupIntentParser().fit(dataset)

        # When / Then
        with self.assertRaises(IntentNotFoundError):
            parser.get_slots("Hello John", "greeting3")

    def test_should_parse_slots_after_deserialization(self):
        # Given
        dataset = self.slots_dataset
        shared = self.get_shared_data(dataset)
        parser = LookupIntentParser(**shared).fit(dataset)
        parser.persist(self.tmp_file_path)
        deserialized_parser = LookupIntentParser.from_path(
            self.tmp_file_path, **shared)

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
- can you prepare [number_of_cups] cup of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        intent_parser = LookupIntentParser(**shared).fit(dataset)

        # When
        intent_parser_bytes = intent_parser.to_byte_array()
        loaded_intent_parser = LookupIntentParser.from_byte_array(
            intent_parser_bytes, **shared)
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
- this is [slot2:entity2](second_entity)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        naughty_strings_path = TEST_PATH / "resources" / "naughty_strings.txt"
        with naughty_strings_path.open(encoding="utf8") as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        # When
        parser = LookupIntentParser().fit(dataset)

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
            {DATA: [{TEXT: naughty_string}]}
            for naughty_string in naughty_strings
        ]

        # When
        naughty_dataset = {
            "intents": {"naughty_intent": {"utterances": utterances}},
            "entities": dict(),
            "language": "en",
        }

        # Then
        with self.fail_if_exception("Exception raised"):
            LookupIntentParser().fit(naughty_dataset)

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
            parser = LookupIntentParser().fit(naughty_dataset)
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
        config = LookupIntentParserConfig(ignore_stop_words=True)
        parser = LookupIntentParser(config=config)

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
            "entity_scopes": None,
            "stop_words_whitelist": None
        }

        metadata = {"unit_name": "lookup_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(
            self.tmp_file_path / "intent_parser.json", expected_dict)

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
  - [paris, city of lights]""")

        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        mock_get_stop_words.return_value = {"a", "me"}
        config = LookupIntentParserConfig(ignore_stop_words=True)
        parser = LookupIntentParser(config=config).fit(dataset)

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
                "-2020846245": [0, [0, 1]],
                "-1558674456": [0, [1]],
            },
            "slots_names": ["origin", "destination"],
            "entity_scopes": [
                {
                    "entity_scope": {"builtin": [], "custom": ["city"]},
                    "intent_group": ["searchFlight"]
                }
            ],
            "stop_words_whitelist": dict()
        }
        metadata = {"unit_name": "lookup_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(
            self.tmp_file_path / "intent_parser.json", expected_dict)

    def test_should_be_deserializable(self):
        # Given
        parser_dict = {
            "config": {
                "unit_name": "lookup_intent_parser",
                "ignore_stop_words": True
            },
            "language_code": "en",
            "map": {
                hash_str("make coffee"): [0, []],
                hash_str("prepare % snipsnumber % coffees"): [0, [0]],
                hash_str("% snipsnumber % teas at % snipstemperature %"):
                    [1, [0, 1]],
            },
            "slots_names": ["nb_cups", "tea_temperature"],
            "intents_names": ["MakeCoffee", "MakeTea"],
            "entity_scopes": [
                {
                    "entity_scope": {
                        "builtin": ["snips/number"],
                        "custom": [],
                    },
                    "intent_group": ["MakeCoffee"]
                },
                {
                    "entity_scope": {
                        "builtin": ["snips/number", "snips/temperature"],
                        "custom": [],
                    },
                    "intent_group": ["MakeTea"]
                },
            ],
            "stop_words_whitelist": dict()
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "lookup_intent_parser"}
        self.writeJsonContent(
            self.tmp_file_path / "intent_parser.json", parser_dict)
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        resources = self.get_resources("en")
        print("STOP WORDS = %s" % resources[STOP_WORDS])
        builtin_entity_parser = BuiltinEntityParser.build(language="en")
        custom_entity_parser = EntityParserMock()

        # When
        parser = LookupIntentParser.from_path(
            self.tmp_file_path, custom_entity_parser=custom_entity_parser,
            builtin_entity_parser=builtin_entity_parser,
            resources=resources)
        res_make_coffee = parser.parse("make me a coffee")
        res_make_tea = parser.parse("two teas at 90°C please")

        # Then
        expected_result_coffee = parsing_result(
            input="make me a coffee",
            intent=intent_classification_result("MakeCoffee", 1.0),
            slots=[])
        expected_result_tea = parsing_result(
            input="two teas at 90°C please",
            intent=intent_classification_result("MakeTea", 1.0),
            slots=[
                {
                    "entity": "snips/number",
                    "range": {"end": 3, "start": 0},
                    "slotName": "nb_cups",
                    "value": "two"
                },
                {
                    "entity": "snips/temperature",
                    "range": {"end": 16, "start": 12},
                    "slotName": "tea_temperature",
                    "value": "90°C"
                }
            ])
        self.assertEqual(expected_result_coffee, res_make_coffee)
        self.assertEqual(expected_result_tea, res_make_tea)

    def test_should_be_deserializable_before_fitting(self):
        # Given
        parser_dict = {
            "config": {},
            "language_code": None,
            "map": None,
            "slots_names": [],
            "intents_names": [],
            "entity_scopes": None
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "dict_deterministic_intent_parser"}
        self.writeJsonContent(
            self.tmp_file_path / "intent_parser.json", parser_dict)
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)

        # When
        parser = LookupIntentParser.from_path(self.tmp_file_path)

        # Then
        config = LookupIntentParserConfig()
        expected_parser = LookupIntentParser(config=config)
        self.assertEqual(parser.to_dict(), expected_parser.to_dict())

    def test_get_entity_scopes(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - meeting [schedule_time:snips/datetime](today)

---
type: intent
name: intent2
utterances:
  - hello world

---
type: intent
name: intent3
utterances:
  - what will be the weather [weather_time:snips/datetime](tomorrow)
  
---
type: intent
name: intent4
utterances:
  - find a flight for [city](Paris) [flight_time:snips/datetime](tomorrow)""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        # When
        entity_scopes = _get_entity_scopes(dataset)

        # Then
        expected_scopes = [
            {
                "entity_scope": {
                    "builtin": ["snips/datetime"],
                    "custom": []
                },
                "intent_group": ["intent1", "intent3"]
            },
            {
                "entity_scope": {
                    "builtin": [],
                    "custom": []
                },
                "intent_group": ["intent2"]
            },
            {
                "entity_scope": {
                    "builtin": ["snips/datetime"],
                    "custom": ["city"]
                },
                "intent_group": ["intent4"]
            }
        ]

        def sort_key(group_scope):
            return " ".join(group_scope["intent_group"])

        self.assertListEqual(sorted(expected_scopes, key=sort_key),
                             sorted(entity_scopes, key=sort_key))
