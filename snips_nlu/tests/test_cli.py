# coding=utf-8
from __future__ import unicode_literals

import shutil
import tempfile

from snips_nlu import SnipsNLUEngine
from snips_nlu.cli import cross_val_metrics, parse, train, train_test_metrics
from snips_nlu.cli.dataset import AssistantDataset
from snips_nlu.cli.dataset.entities import CustomEntity
from snips_nlu.cli.dataset.intent_dataset import IntentDataset
from snips_nlu.constants import PACKAGE_PATH
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.tests.utils import BEVERAGE_DATASET_PATH, SnipsTest, TEST_PATH


def mk_sys_argv(args):
    return ["program_name"] + args


class TestCLI(SnipsTest):
    fixture_dir = TEST_PATH / "cli_fixture"

    # pylint: disable=protected-access
    def setUp(self):
        if not self.fixture_dir.exists():
            self.fixture_dir.mkdir()

        self.tmp_file_path = self.fixture_dir / next(
            tempfile._get_candidate_names())
        while self.tmp_file_path.exists():
            self.tmp_file_path = self.fixture_dir / next(
                tempfile._get_candidate_names())

    def tearDown(self):
        if self.fixture_dir.exists():
            shutil.rmtree(str(self.fixture_dir))

    def test_train(self):
        # Given / When
        train(BEVERAGE_DATASET_PATH, str(self.tmp_file_path), config_path=None)

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No trained engine generated")
        msg = "Failed to create an engine from engine dict."
        with self.fail_if_exception(msg):
            SnipsNLUEngine.from_path(self.tmp_file_path)

    def test_parse(self):
        # Given / When
        train(BEVERAGE_DATASET_PATH, str(self.tmp_file_path), config_path=None)

        # When
        with self.fail_if_exception("Failed to parse using CLI script"):
            parse(str(self.tmp_file_path), "Make me two cups of coffee")

    def test_cross_val_metrics(self):
        # Given / When
        cross_val_metrics(str(BEVERAGE_DATASET_PATH), str(self.tmp_file_path))

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No metrics found")

    def test_train_test_metrics(self):
        # Given / When
        train_test_metrics(str(BEVERAGE_DATASET_PATH),
                           str(BEVERAGE_DATASET_PATH), str(self.tmp_file_path))

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No metrics found")

    def test_should_generate_intent_from_file(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        intent_file = examples_path / "intent_getWeather.txt"

        # When
        intent_dataset = IntentDataset.from_file(intent_file)
        intent_dict = intent_dataset.json

        # Then
        expected_intent_dict = {
            "utterances": [
                {
                    "data": [
                        {
                            "text": "what is the weather in "
                        },
                        {
                            "entity": "location",
                            "slot_name": "weatherLocation",
                            "text": "Paris"
                        },
                        {
                            "text": "?"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "Will it rain "
                        },
                        {
                            "entity": "snips/datetime",
                            "slot_name": "weatherDate",
                            "text": "tomorrow"
                        },
                        {
                            "text": " in "
                        },
                        {
                            "entity": "location",
                            "slot_name": "weatherLocation",
                            "text": "Moscow"
                        },
                        {
                            "text": "?"
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "text": "How is the weather in "
                        },
                        {
                            "entity": "location",
                            "slot_name": "weatherLocation",
                            "text": "San Francisco"
                        },
                        {
                            "entity": "snips/datetime",
                            "slot_name": "weatherDate",
                            "text": "today"
                        },
                        {
                            "text": "?"
                        }
                    ]
                }
            ]
        }

        self.assertDictEqual(expected_intent_dict, intent_dict)

    def test_should_generate_entity_from_file(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        entity_file = examples_path / "entity_location.txt"

        # When
        entity_dataset = CustomEntity.from_file(entity_file)
        entity_dict = entity_dataset.json

        # Then
        expected_entity_dict = {
            "automatically_extensible": True,
            "data": [
                {
                    "synonyms": [
                        "big apple"
                    ],
                    "value": "new york"
                },
                {
                    "synonyms": [
                        "city of lights"
                    ],
                    "value": "paris"
                },
                {
                    "synonyms": [],
                    "value": "london"
                }
            ],
            "use_synonyms": True
        }
        self.assertDictEqual(expected_entity_dict, entity_dict)

    def test_should_generate_entity_from_file_with_autoextensible(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        entity_file = examples_path / "entity_location_autoextent_false.txt"

        # When
        entity_dataset = CustomEntity.from_file(entity_file)
        entity_dict = entity_dataset.json

        # Then
        expected_entity_dict = {
            "automatically_extensible": False,
            "data": [
                {
                    "synonyms": [
                        "big apple"
                    ],
                    "value": "new york"
                },
                {
                    "synonyms": [
                        "city of lights"
                    ],
                    "value": "paris"
                },
                {
                    "synonyms": [],
                    "value": "london"
                }
            ],
            "use_synonyms": True
        }
        self.assertDictEqual(expected_entity_dict, entity_dict)

    def test_should_generate_dataset_from_files(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        intent_file_1 = examples_path / "intent_whoIsGame.txt"
        intent_file_2 = examples_path / "intent_getWeather.txt"
        entity_file_1 = examples_path / "entity_location.txt"

        dataset = AssistantDataset.from_files(
            "en", [intent_file_1, intent_file_2, entity_file_1])
        dataset_dict = dataset.json

        # When / Then
        expected_dataset_dict = {
            "entities": {
                "company": {
                    "automatically_extensible": True,
                    "data": [],
                    "use_synonyms": True
                },
                "country": {
                    "automatically_extensible": True,
                    "data": [],
                    "use_synonyms": True
                },
                "location": {
                    "automatically_extensible": True,
                    "data": [
                        {
                            "synonyms": [
                                "big apple"
                            ],
                            "value": "new york"
                        },
                        {
                            "synonyms": [
                                "city of lights"
                            ],
                            "value": "paris"
                        },
                        {
                            "synonyms": [],
                            "value": "london"
                        }
                    ],
                    "use_synonyms": True
                },
                "role": {
                    "automatically_extensible": True,
                    "data": [],
                    "use_synonyms": True
                },
                "snips/datetime": {}
            },
            "intents": {
                "getWeather": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "what is the weather in "
                                },
                                {
                                    "entity": "location",
                                    "slot_name": "weatherLocation",
                                    "text": "Paris"
                                },
                                {
                                    "text": "?"
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "Will it rain "
                                },
                                {
                                    "entity": "snips/datetime",
                                    "slot_name": "weatherDate",
                                    "text": "tomorrow"
                                },
                                {
                                    "text": " in "
                                },
                                {
                                    "entity": "location",
                                    "slot_name": "weatherLocation",
                                    "text": "Moscow"
                                },
                                {
                                    "text": "?"
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "How is the weather in "
                                },
                                {
                                    "entity": "location",
                                    "slot_name": "weatherLocation",
                                    "text": "San Francisco"
                                },
                                {
                                    "entity": "snips/datetime",
                                    "slot_name": "weatherDate",
                                    "text": "today"
                                },
                                {
                                    "text": "?"
                                }
                            ]
                        }
                    ]
                },
                "whoIsGame": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "who is the "
                                },
                                {
                                    "entity": "role",
                                    "slot_name": "role",
                                    "text": "president"
                                },
                                {
                                    "text": " of "
                                },
                                {
                                    "entity": "country",
                                    "slot_name": "country",
                                    "text": "France"
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "who is the "
                                },
                                {
                                    "entity": "role",
                                    "slot_name": "role",
                                    "text": "prime minister"
                                },
                                {
                                    "text": " of "
                                },
                                {
                                    "entity": "country",
                                    "slot_name": "country",
                                    "text": "UK"
                                }
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "who is the "
                                },
                                {
                                    "entity": "role",
                                    "slot_name": "role",
                                    "text": "CEO"
                                },
                                {
                                    "text": " of "
                                },
                                {
                                    "entity": "company",
                                    "slot_name": "company",
                                    "text": "Google"
                                },
                                {
                                    "text": " please"
                                }
                            ]
                        }
                    ]
                }
            },
            "language": "en"
        }
        validate_and_format_dataset(dataset_dict)
        self.assertDictEqual(expected_dataset_dict, dataset_dict)

    def test_should_fail_generating_intent_with_wrong_file_name(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        intent_file = examples_path / "getWeather.txt"

        # When / Then
        with self.assertRaises(AssertionError):
            IntentDataset.from_file(intent_file)

    def test_should_fail_generating_entity_with_wrong_file_name(self):
        # Given
        examples_path = PACKAGE_PATH / "cli" / "dataset" / "examples"
        entity_file = examples_path / "location.txt"

        # When / Then
        with self.assertRaises(AssertionError):
            CustomEntity.from_file(entity_file)
