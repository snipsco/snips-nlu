# coding=utf-8
from __future__ import unicode_literals

from builtins import range
from mock import patch

from snips_nlu.constants import (
    DATA, END, ENTITY, ENTITY_KIND, LANGUAGE_EN, RES_INTENT, RES_INTENT_NAME,
    RES_MATCH_RANGE, RES_SLOTS, SLOT_NAME, SNIPS_DATETIME, SNIPS_ORDINAL,
    START, TEXT, VALUE)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.deterministic_intent_parser import (
    DeterministicIntentParser, _deduplicate_overlapping_slots,
    _get_range_shift, _replace_builtin_entities,
    _replace_tokenized_out_characters)
from snips_nlu.pipeline.configs import DeterministicIntentParserConfig
from snips_nlu.result import intent_classification_result, unresolved_slot
from snips_nlu.tests.utils import FixtureTest, SAMPLE_DATASET, TEST_PATH


class TestDeterministicIntentParser(FixtureTest):
    def setUp(self):
        super(TestDeterministicIntentParser, self).setUp()
        self.duplicated_utterances_dataset = {
            "entities": {},
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "Hello world"
                                }
                            ]
                        }
                    ]
                },
                "dummy_intent_2": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "Hello world"
                                }
                            ]
                        }
                    ]
                }
            },
            "language": "en",
            "snips_nlu_version": "0.1.0"
        }

        self.slots_dataset = {
            "entities": {
                "dummy_entity_1": {
                    "automatically_extensible": False,
                    "use_synonyms": True,
                    "data": [
                        {
                            "synonyms": [
                                "dummy_a",
                                "dummy 2a",
                                "dummy a",
                                "2 dummy a"
                            ],
                            "value": "dummy_a"
                        },
                        {
                            "synonyms": [
                                "dummy_b",
                                "dummy_bb",
                                "dummy b"
                            ],
                            "value": "dummy_b"
                        },
                        {
                            "synonyms": [
                                "dummy d"
                            ],
                            "value": "dummy d"
                        }
                    ]
                },
                "dummy_entity_2": {
                    "automatically_extensible": False,
                    "use_synonyms": True,
                    "data": [
                        {
                            "synonyms": [
                                "dummy_c",
                                "dummy_cc",
                                "dummy c",
                                "3p.m."
                            ],
                            "value": "dummy_c"
                        }
                    ]
                },
                "snips/datetime": {}
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "This is a "
                                },
                                {
                                    "text": "dummy_1",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                },
                                {
                                    "text": " query with another "
                                },
                                {
                                    "text": "dummy_2",
                                    "slot_name": "dummy_slot_name2",
                                    "entity": "dummy_entity_2"
                                },
                                {
                                    "text": " "
                                },
                                {
                                    "text": "at 10p.m.",
                                    "slot_name": "startTime",
                                    "entity": "snips/datetime"
                                },
                                {
                                    "text": " or "
                                },
                                {
                                    "text": "next monday",
                                    "slot_name": "startTime",
                                    "entity": "snips/datetime"
                                }

                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "This    is  a  "
                                },
                                {
                                    "text": "dummy_1",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                },
                                {
                                    "text": " "
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "tomorrow evening",
                                    "slot_name": "startTime",
                                    "entity": "snips/datetime"
                                },
                                {
                                    "text": " there is a "
                                },
                                {
                                    "text": "dummy_1",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                }
                            ]
                        },
                    ]
                }
            },
            "language": "en",
            "snips_nlu_version": "1.0.1"
        }

    def test_should_get_intent(self):
        # Given
        dataset = validate_and_format_dataset(self.slots_dataset)

        parser = DeterministicIntentParser().fit(dataset)
        text = "this is a dummy_a query with another dummy_c at 10p.m. or " \
               "at 12p.m."

        # When
        parsing = parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability)

        self.assertEqual(expected_intent, parsing[RES_INTENT])

    def test_should_get_intent_when_filter(self):
        # Given
        dataset = validate_and_format_dataset(
            self.duplicated_utterances_dataset)

        parser = DeterministicIntentParser().fit(dataset)
        text = "Hello world"
        intent_name_1 = "dummy_intent_1"
        intent_name_2 = "dummy_intent_2"

        # When
        res_1 = parser.parse(text, intent_name_1)
        res_2 = parser.parse(text, [intent_name_2])

        # Then
        self.assertEqual(intent_name_1, res_1[RES_INTENT][RES_INTENT_NAME])
        self.assertEqual(intent_name_2, res_2[RES_INTENT][RES_INTENT_NAME])

    def test_should_get_intent_after_deserialization(self):
        # Given
        dataset = validate_and_format_dataset(self.slots_dataset)

        parser = DeterministicIntentParser().fit(dataset)
        parser.persist(self.tmp_file_path)
        deserialized_parser = DeterministicIntentParser.from_path(
            self.tmp_file_path)
        text = "this is a dummy_a query with another dummy_c at 10p.m. or " \
               "at 12p.m."

        # When
        parsing = deserialized_parser.parse(text)

        # Then
        probability = 1.0
        expected_intent = intent_classification_result(
            intent_name="dummy_intent_1", probability=probability)
        self.assertEqual(expected_intent, parsing[RES_INTENT])

    def test_should_get_slots(self):
        # Given
        dataset = self.slots_dataset
        dataset = validate_and_format_dataset(dataset)

        parser = DeterministicIntentParser().fit(dataset)
        texts = [
            (
                "this is a dummy a query with another dummy_c at 10p.m. or at"
                " 12p.m.",
                [
                    unresolved_slot(match_range=(10, 17), value="dummy a",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name"),
                    unresolved_slot(match_range=(37, 44), value="dummy_c",
                                    entity="dummy_entity_2",
                                    slot_name="dummy_slot_name2"),
                    unresolved_slot(match_range=(45, 54), value="at 10p.m.",
                                    entity="snips/datetime",
                                    slot_name="startTime"),
                    unresolved_slot(match_range=(58, 67), value="at 12p.m.",
                                    entity="snips/datetime",
                                    slot_name="startTime")
                ]
            ),
            (
                "this, is,, a, dummy a query with another dummy_c at 10pm or "
                "at 12p.m.",
                [
                    unresolved_slot(match_range=(14, 21), value="dummy a",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name"),
                    unresolved_slot(match_range=(41, 48), value="dummy_c",
                                    entity="dummy_entity_2",
                                    slot_name="dummy_slot_name2"),
                    unresolved_slot(match_range=(49, 56), value="at 10pm",
                                    entity="snips/datetime",
                                    slot_name="startTime"),
                    unresolved_slot(match_range=(60, 69), value="at 12p.m.",
                                    entity="snips/datetime",
                                    slot_name="startTime")
                ]
            ),
            (
                "this is a dummy b",
                [
                    unresolved_slot(match_range=(10, 17), value="dummy b",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name")
                ]
            ),
            (
                " this is a dummy b ",
                [
                    unresolved_slot(match_range=(11, 18), value="dummy b",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name")
                ]
            ),
            (
                " at 8am ’ there is a dummy  a",
                [
                    unresolved_slot(match_range=(1, 7), value="at 8am",
                                    entity="snips/datetime",
                                    slot_name="startTime"),
                    unresolved_slot(match_range=(21, 29), value="dummy  a",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name")
                ]
            )
        ]

        for text, expected_slots in texts:
            # When
            parsing = parser.parse(text)

            # Then
            self.assertListEqual(expected_slots, parsing[RES_SLOTS])

    def test_should_get_slots_after_deserialization(self):
        # Given
        dataset = self.slots_dataset
        dataset = validate_and_format_dataset(dataset)

        parser = DeterministicIntentParser().fit(dataset)
        parser.persist(self.tmp_file_path)
        deserialized_parser = DeterministicIntentParser.from_path(
            self.tmp_file_path)
        texts = [
            (
                "this is a dummy a query with another dummy_c at 10p.m. or at"
                " 12p.m.",
                [
                    unresolved_slot(match_range=(10, 17), value="dummy a",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name"),
                    unresolved_slot(match_range=(37, 44), value="dummy_c",
                                    entity="dummy_entity_2",
                                    slot_name="dummy_slot_name2"),
                    unresolved_slot(match_range=(45, 54), value="at 10p.m.",
                                    entity="snips/datetime",
                                    slot_name="startTime"),
                    unresolved_slot(match_range=(58, 67), value="at 12p.m.",
                                    entity="snips/datetime",
                                    slot_name="startTime")
                ]
            ),
            (
                "this, is,, a, dummy a query with another dummy_c at 10pm or "
                "at 12p.m.",
                [
                    unresolved_slot(match_range=(14, 21), value="dummy a",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name"),
                    unresolved_slot(match_range=(41, 48), value="dummy_c",
                                    entity="dummy_entity_2",
                                    slot_name="dummy_slot_name2"),
                    unresolved_slot(match_range=(49, 56), value="at 10pm",
                                    entity="snips/datetime",
                                    slot_name="startTime"),
                    unresolved_slot(match_range=(60, 69), value="at 12p.m.",
                                    entity="snips/datetime",
                                    slot_name="startTime")
                ]
            ),
            (
                "this is a dummy b",
                [
                    unresolved_slot(match_range=(10, 17), value="dummy b",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name")
                ]
            ),
            (
                " this is a dummy b ",
                [
                    unresolved_slot(match_range=(11, 18), value="dummy b",
                                    entity="dummy_entity_1",
                                    slot_name="dummy_slot_name")
                ]
            )
        ]

        for text, expected_slots in texts:
            # When
            parsing = deserialized_parser.parse(text)

            # Then
            self.assertListEqual(expected_slots, parsing[RES_SLOTS])

    def test_should_parse_naughty_strings(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        naughty_strings_path = TEST_PATH / "resources" / "naughty_strings.txt"
        with naughty_strings_path.open(encoding='utf8') as f:
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
            "snips_nlu_version": "0.0.1"
        }

        # Then
        with self.fail_if_exception("Exception raised"):
            DeterministicIntentParser().fit(naughty_dataset)

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
                    "data": []
                }
            },
            "language": "en",
            "snips_nlu_version": "0.0.1"
        }

        naughty_dataset = validate_and_format_dataset(naughty_dataset)

        # Then
        with self.fail_if_exception("Exception raised"):
            parser = DeterministicIntentParser()
            parser.fit(naughty_dataset)
            parsing = parser.parse("string0")

            expected_slot = {
                'entity': 'non_ascìi_entïty',
                'range': {
                    "start": 0,
                    "end": 7
                },
                'slotName': u'non_ascìi_slöt',
                'value': u'string0'
            }
            intent_name = parsing[RES_INTENT][RES_INTENT_NAME]
            self.assertEqual("naughty_intent", intent_name)
            self.assertListEqual([expected_slot], parsing[RES_SLOTS])

    def test_should_be_serializable_before_fitting(self):
        # Given
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_pattern_length=43)
        parser = DeterministicIntentParser(config=config)

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_dict = {
            "unit_name": "deterministic_intent_parser",
            "config": {
                "unit_name": "deterministic_intent_parser",
                "max_queries": 42,
                "max_pattern_length": 43
            },
            "language_code": None,
            "group_names_to_slot_names": None,
            "patterns": None,
            "slot_names_to_entities": None
        }

        metadata = {"unit_name": "deterministic_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json",
                               metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json",
                               expected_dict)

    @patch("snips_nlu.intent_parser.deterministic_intent_parser"
           "._generate_patterns")
    def test_should_be_serializable(self, mocked_generate_regexes):
        # Given

        # pylint: disable=unused-argument
        def mock_generate_patterns(utterances, joined_entity_utterances,
                                   group_names_to_slot_names, language):
            patterns = ["mocked_regex_%s" % i for i in range(len(utterances))]
            group_to_slot = {"group_0": "dummy slot name"}
            return patterns, group_to_slot

        # pylint: enable=unused-argument

        mocked_generate_regexes.side_effect = mock_generate_patterns
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_pattern_length=100)
        parser = DeterministicIntentParser(config=config).fit(dataset)

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_dict = {
            "unit_name": "deterministic_intent_parser",
            "config": {
                "unit_name": "deterministic_intent_parser",
                "max_queries": 42,
                "max_pattern_length": 100
            },
            "language_code": "en",
            "group_names_to_slot_names": {
                "group_0": "dummy slot name"
            },
            "patterns": {
                "dummy_intent_1": [
                    "mocked_regex_0",
                    "mocked_regex_1",
                    "mocked_regex_2",
                    "mocked_regex_3"
                ],
                "dummy_intent_2": [
                    "mocked_regex_0"
                ]
            },
            "slot_names_to_entities": {
                "dummy_intent_1": {
                    "dummy_slot_name": "dummy_entity_1",
                    "dummy_slot_name3": "dummy_entity_2",
                    "dummy_slot_name2": "dummy_entity_2"
                },
                "dummy_intent_2": {
                    "dummy slot nàme": "dummy_entity_1"
                }
            }
        }
        metadata = {"unit_name": "deterministic_intent_parser"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json",
                               metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_parser.json",
                               expected_dict)

    def test_should_be_deserializable(self):
        # Given
        parser_dict = {
            "config": {
                "max_queries": 42,
                "max_pattern_length": 43
            },
            "language_code": "en",
            "group_names_to_slot_names": {
                "hello_group": "hello_slot",
                "world_group": "world_slot"
            },
            "patterns": {
                "intent_name": [
                    "(?P<hello_group>hello?)",
                    "(?P<world_group>world$)"
                ]
            },
            "slot_names_to_entities": {
                "hello_slot": "hello_entity",
                "world_slot": "world_entity"
            }
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "deterministic_intent_parser"}
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json",
                              parser_dict)
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)

        # When
        parser = DeterministicIntentParser.from_path(self.tmp_file_path)

        # Then
        patterns = {
            "intent_name": [
                "(?P<hello_group>hello?)",
                "(?P<world_group>world$)"
            ]
        }
        group_names_to_slot_names = {
            "hello_group": "hello_slot",
            "world_group": "world_slot"
        }
        slot_names_to_entities = {
            "hello_slot": "hello_entity",
            "world_slot": "world_entity"
        }
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_pattern_length=43)
        expected_parser = DeterministicIntentParser(config=config)
        expected_parser.language = LANGUAGE_EN
        expected_parser.group_names_to_slot_names = group_names_to_slot_names
        expected_parser.slot_names_to_entities = slot_names_to_entities
        expected_parser.patterns = patterns

        self.assertEqual(parser.to_dict(), expected_parser.to_dict())

    def test_should_be_deserializable_before_fitting(self):
        # Given
        parser_dict = {
            "config": {
                "max_queries": 42,
                "max_pattern_length": 43
            },
            "language_code": None,
            "group_names_to_slot_names": None,
            "patterns": None,
            "slot_names_to_entities": None
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "deterministic_intent_parser"}
        self.writeJsonContent(self.tmp_file_path / "intent_parser.json",
                              parser_dict)
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)

        # When
        parser = DeterministicIntentParser.from_path(self.tmp_file_path)

        # Then
        config = DeterministicIntentParserConfig(max_queries=42,
                                                 max_pattern_length=43)
        expected_parser = DeterministicIntentParser(config=config)
        self.assertEqual(parser.to_dict(), expected_parser.to_dict())

    def test_should_deduplicate_overlapping_slots(self):
        # Given
        language = LANGUAGE_EN
        slots = [
            unresolved_slot(
                [3, 7],
                "non_overlapping1",
                "e",
                "s1"
            ),
            unresolved_slot(
                [9, 16],
                "aaaaaaa",
                "e1",
                "s2"
            ),
            unresolved_slot(
                [10, 18],
                "bbbbbbbb",
                "e1",
                "s3"
            ),
            unresolved_slot(
                [17, 23],
                "b cccc",
                "e1",
                "s4"
            ),
            unresolved_slot(
                [50, 60],
                "non_overlapping2",
                "e",
                "s5"
            ),
        ]

        # When
        deduplicated_slots = _deduplicate_overlapping_slots(slots, language)

        # Then
        expected_slots = [
            unresolved_slot(
                [3, 7],
                "non_overlapping1",
                "e",
                "s1"
            ),
            unresolved_slot(
                [17, 23],
                "b cccc",
                "e1",
                "s4"
            ),
            unresolved_slot(
                [50, 60],
                "non_overlapping2",
                "e",
                "s5"
            ),
        ]
        self.assertSequenceEqual(deduplicated_slots, expected_slots)

    def test_should_limit_nb_queries(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        config = DeterministicIntentParserConfig(max_queries=2,
                                                 max_pattern_length=1000)

        # When
        parser = DeterministicIntentParser(config=config).fit(dataset)

        # Then
        self.assertEqual(len(parser.regexes_per_intent["dummy_intent_1"]), 2)
        self.assertEqual(len(parser.regexes_per_intent["dummy_intent_2"]), 1)

    def test_should_limit_patterns_length(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        config = DeterministicIntentParserConfig(max_queries=1000,
                                                 max_pattern_length=300)

        # When
        parser = DeterministicIntentParser(config=config).fit(dataset)

        # Then
        self.assertEqual(3, len(parser.regexes_per_intent["dummy_intent_1"]))
        self.assertEqual(1, len(parser.regexes_per_intent["dummy_intent_2"]))

    @patch('snips_nlu.intent_parser.deterministic_intent_parser'
           '.get_builtin_entities')
    def test_should_replace_builtin_entities(self, mock_get_builtin_entities):
        # Given
        text = "Be the first to be there at 9pm"
        mock_get_builtin_entities.return_value = [
            {
                RES_MATCH_RANGE: {START: 7, END: 12},
                VALUE: "first",
                ENTITY_KIND: SNIPS_ORDINAL
            },
            {
                RES_MATCH_RANGE: {START: 28, END: 31},
                VALUE: "9pm",
                ENTITY_KIND: SNIPS_DATETIME
            }
        ]

        # When
        range_mapping, processed_text = _replace_builtin_entities(
            text=text, language=LANGUAGE_EN)

        # Then
        expected_mapping = {
            (7, 21): {START: 7, END: 12},
            (37, 52): {START: 28, END: 31}
        }
        expected_processed_text = \
            "Be the %SNIPSORDINAL% to be there at %SNIPSDATETIME%"

        self.assertDictEqual(expected_mapping, range_mapping)
        self.assertEqual(expected_processed_text, processed_text)

    def test_should_get_range_shift(self):
        # Given
        ranges_mapping = {
            (2, 5): {START: 2, END: 4},
            (8, 9): {START: 7, END: 11}
        }

        # When / Then
        self.assertEqual(-1, _get_range_shift((6, 7), ranges_mapping))
        self.assertEqual(2, _get_range_shift((12, 13), ranges_mapping))

    def test_should_replace_tokenized_out_characters(self):
        # Given
        string = ": hello, it's me !  "

        # When
        cleaned_string = _replace_tokenized_out_characters(string, "en", "_")

        # Then
        self.assertEqual("__hello__it_s_me_!__", cleaned_string)
