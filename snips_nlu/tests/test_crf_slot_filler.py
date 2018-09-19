# coding=utf-8
from __future__ import unicode_literals

from builtins import range
from pathlib import Path

from mock import MagicMock

from snips_nlu.constants import (
    DATA, END, ENTITY, ENTITY_KIND, LANGUAGE_EN, RES_MATCH_RANGE, SLOT_NAME,
    SNIPS_DATETIME, START, TEXT, VALUE)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.pipeline.configs import CRFSlotFillerConfig
from snips_nlu.preprocessing import Token, tokenize
from snips_nlu.result import unresolved_slot
from snips_nlu.slot_filler.crf_slot_filler import (
    CRFSlotFiller, _disambiguate_builtin_entities,
    _filter_overlapping_builtins, _get_slots_permutations,
    _spans_to_tokens_indexes)
from snips_nlu.slot_filler.crf_utils import (
    BEGINNING_PREFIX, INSIDE_PREFIX, TaggingScheme)
from snips_nlu.slot_filler.feature_factory import (
    IsDigitFactory, NgramFactory, ShapeNgramFactory)
from snips_nlu.tests.utils import (
    BEVERAGE_DATASET, FixtureTest, SAMPLE_DATASET, TEST_PATH, WEATHER_DATASET)


class TestCRFSlotFiller(FixtureTest):
    def test_should_get_slots(self):
        # Given
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        config = CRFSlotFillerConfig(random_seed=42)
        intent = "MakeTea"
        slot_filler = CRFSlotFiller(config)
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
        dataset = validate_and_format_dataset(WEATHER_DATASET)
        config = CRFSlotFillerConfig(random_seed=42)
        intent = "SearchWeatherForecast"
        slot_filler = CRFSlotFiller(config)
        slot_filler.fit(dataset, intent)

        # When
        slots = slot_filler.get_slots("Give me the weather at 9p.m. in Paris")

        # Then
        expected_slots = [
            unresolved_slot(match_range={START: 20, END: 28},
                            value='at 9p.m.',
                            entity='snips/datetime',
                            slot_name='datetime'),
            unresolved_slot(match_range={START: 32, END: 37},
                            value='Paris',
                            entity='weather_location',
                            slot_name='location')
        ]
        self.assertListEqual(expected_slots, slots)

    def test_should_parse_naughty_strings(self):
        # Given
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        naughty_strings_path = TEST_PATH / "resources" / "naughty_strings.txt"
        with naughty_strings_path.open(encoding='utf8') as f:
            naughty_strings = [line.strip("\n") for line in f.readlines()]

        # When
        slot_filler = CRFSlotFiller().fit(dataset, "dummy_intent_1")

        # Then
        for s in naughty_strings:
            with self.fail_if_exception("Naughty string crashes"):
                slot_filler.get_slots(s)

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
        with self.fail_if_exception("Naughty string crashes"):
            CRFSlotFiller().fit(naughty_dataset, "naughty_intent")

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
        with self.fail_if_exception("Naughty string make NLU crash"):
            slot_filler = CRFSlotFiller()
            slot_filler.fit(naughty_dataset, "naughty_intent")
            slots = slot_filler.get_slots("string0")
            expected_slot = {
                "entity": "non_ascìi_entïty",
                "probability": 1.0,
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
        dataset = validate_and_format_dataset(BEVERAGE_DATASET)
        config = CRFSlotFillerConfig(random_seed=42)
        intent = "MakeTea"
        slot_filler = CRFSlotFiller(config)
        slot_filler.fit(dataset, intent)
        slot_filler.persist(self.tmp_file_path)
        deserialized_slot_filler = CRFSlotFiller.from_path(self.tmp_file_path)

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
            "unit_name": "crf_slot_filler",
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
        dataset = validate_and_format_dataset(SAMPLE_DATASET)

        slot_filler = CRFSlotFiller(config)
        intent = "dummy_intent_1"
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
            "unit_name": "crf_slot_filler",
            "crf_model_file": expected_crf_file,
            "language_code": "en",
            "config": expected_config.to_dict(),
            "intent": intent,
            "slot_name_mapping": {
                "dummy_slot_name": "dummy_entity_1",
                "dummy_slot_name2": "dummy_entity_2",
                "dummy_slot_name3": "dummy_entity_2",
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
            feature_factory_configs=features_factories, random_seed=40)
        slot_filler = CRFSlotFiller(slot_filler_config)

        tokens = tokenize("foo hello world bar", LANGUAGE_EN)
        dataset = validate_and_format_dataset(SAMPLE_DATASET)
        slot_filler.fit(dataset, intent="dummy_intent_1")

        # When
        features_with_drop_out = slot_filler.compute_features(tokens, True)

        # Then
        expected_features = [
            {"ngram_1": "foo"},
            {},
            {"ngram_1": "world"},
            {},
        ]
        self.assertListEqual(expected_features, features_with_drop_out)

    def test_spans_to_tokens_indexes(self):
        # Given
        spans = [
            {START: 0, END: 1},
            {START: 2, END: 6},
            {START: 5, END: 6},
            {START: 9, END: 15}
        ]
        tokens = [
            Token(value="abc", start=0, end=3),
            Token(value="def", start=4, end=7),
            Token(value="ghi", start=10, end=13)
        ]

        # When
        indexes = _spans_to_tokens_indexes(spans, tokens)

        # Then
        expected_indexes = [[0], [0, 1], [1], [2]]
        self.assertListEqual(indexes, expected_indexes)

    def test_augment_slots(self):
        # Given
        language = LANGUAGE_EN
        text = "Find me a flight before 10pm and after 8pm"
        tokens = tokenize(text, language)
        missing_slots = {"start_date", "end_date"}

        tags = ['O' for _ in tokens]

        def mocked_sequence_probability(_, tags_):
            tags_1 = ['O',
                      'O',
                      'O',
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX,
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX]

            tags_2 = ['O',
                      'O',
                      'O',
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX,
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX]

            tags_3 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

            tags_4 = ['O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX]

            tags_5 = ['O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX]

            tags_6 = ['O',
                      'O',
                      'O',
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX,
                      'O',
                      'O',
                      'O']

            tags_7 = ['O',
                      'O',
                      'O',
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX,
                      'O',
                      'O',
                      'O']

            tags_8 = ['O',
                      'O',
                      'O',
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX,
                      'O',
                      '%sstart_date' % BEGINNING_PREFIX,
                      '%sstart_date' % INSIDE_PREFIX]

            tags_9 = ['O',
                      'O',
                      'O',
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX,
                      'O',
                      '%send_date' % BEGINNING_PREFIX,
                      '%send_date' % INSIDE_PREFIX]

            if tags_ == tags_1:
                return 0.6
            elif tags_ == tags_2:
                return 0.8
            elif tags_ == tags_3:
                return 0.2
            elif tags_ == tags_4:
                return 0.2
            elif tags_ == tags_5:
                return 0.99
            elif tags_ == tags_6:
                return 0.0
            elif tags_ == tags_7:
                return 0.0
            elif tags_ == tags_8:
                return 0.5
            elif tags_ == tags_9:
                return 0.5
            else:
                raise ValueError("Unexpected tag sequence: %s" % tags_)

        slot_filler_config = CRFSlotFillerConfig(random_seed=42)
        slot_filler = CRFSlotFiller(config=slot_filler_config)
        slot_filler.language = LANGUAGE_EN
        slot_filler.intent = "intent1"
        slot_filler.slot_name_mapping = {
            "start_date": "snips/datetime",
            "end_date": "snips/datetime",
        }

        # pylint:disable=protected-access
        slot_filler._get_sequence_probability = MagicMock(
            side_effect=mocked_sequence_probability)
        # pylint:enable=protected-access

        slot_filler.compute_features = MagicMock(return_value=None)

        # When
        # pylint: disable=protected-access
        augmented_slots = slot_filler._augment_slots(text, tokens, tags,
                                                     missing_slots)
        # pylint: enable=protected-access

        # Then
        expected_slots = [
            unresolved_slot(value='after 8pm',
                            match_range={START: 33, END: 42},
                            entity='snips/datetime', slot_name='end_date')
        ]
        self.assertListEqual(augmented_slots, expected_slots)

    def test_filter_overlapping_builtins(self):
        # Given
        language = LANGUAGE_EN
        text = "Find me a flight before 10pm and after 8pm"
        tokens = tokenize(text, language)
        tags = ['O' for _ in range(5)] + ['B-flight'] + ['O' for _ in range(3)]
        tagging_scheme = TaggingScheme.BIO
        builtin_entities = [
            {
                RES_MATCH_RANGE: {START: 17, END: 28},
                VALUE: "before 10pm",
                ENTITY_KIND: SNIPS_DATETIME
            },
            {
                RES_MATCH_RANGE: {START: 33, END: 42},
                VALUE: "after 8pm",
                ENTITY_KIND: SNIPS_DATETIME
            }
        ]

        # When
        entities = _filter_overlapping_builtins(builtin_entities, tokens, tags,
                                                tagging_scheme)

        # Then
        expected_entities = [
            {
                RES_MATCH_RANGE: {START: 33, END: 42},
                VALUE: "after 8pm",
                ENTITY_KIND: SNIPS_DATETIME
            }
        ]
        self.assertEqual(entities, expected_entities)

    def test_should_disambiguate_builtin_entities(self):
        # Given
        builtin_entities = [
            {RES_MATCH_RANGE: {START: 7, END: 10}},
            {RES_MATCH_RANGE: {START: 9, END: 15}},
            {RES_MATCH_RANGE: {START: 10, END: 17}},
            {RES_MATCH_RANGE: {START: 12, END: 19}},
            {RES_MATCH_RANGE: {START: 9, END: 15}},
            {RES_MATCH_RANGE: {START: 0, END: 5}},
            {RES_MATCH_RANGE: {START: 0, END: 5}},
            {RES_MATCH_RANGE: {START: 0, END: 8}},
            {RES_MATCH_RANGE: {START: 2, END: 5}},
            {RES_MATCH_RANGE: {START: 0, END: 8}},
        ]

        # When
        disambiguated_entities = _disambiguate_builtin_entities(
            builtin_entities)

        # Then
        expected_entities = [
            {RES_MATCH_RANGE: {START: 0, END: 8}},
            {RES_MATCH_RANGE: {START: 0, END: 8}},
            {RES_MATCH_RANGE: {START: 10, END: 17}},
        ]

        self.assertListEqual(expected_entities, disambiguated_entities)

    def test_generate_slots_permutations(self):
        # Given
        slot_name_mapping = {
            "start_date": "snips/datetime",
            "end_date": "snips/datetime",
            "temperature": "snips/temperature"
        }
        grouped_entities = [
            [
                {ENTITY_KIND: "snips/datetime"},
                {ENTITY_KIND: "snips/temperature"}
            ],
            [
                {ENTITY_KIND: "snips/temperature"}
            ]
        ]

        # When
        slots_permutations = set(
            "||".join(perm) for perm in
            _get_slots_permutations(grouped_entities, slot_name_mapping))

        # Then
        expected_permutations = {
            "start_date||temperature",
            "end_date||temperature",
            "temperature||temperature",
            "O||temperature",
            "start_date||O",
            "end_date||O",
            "temperature||O",
            "O||O",
        }
        self.assertSetEqual(expected_permutations, slots_permutations)
