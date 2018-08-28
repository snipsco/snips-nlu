# coding=utf-8
from __future__ import unicode_literals

from future.builtins import str
from mock import patch
from pathlib import Path

from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.entity_parser import CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser import \
    CustomEntityParserUsage
from snips_nlu.tests.utils import FixtureTest

DATASET = {
    "intents": {

    },
    "entities": {
        "dummy_entity_1": {
            "data": [
                {
                    "value": "dummy_entity_1",
                    "synonyms": ["dummy_1"]
                }
            ],
            "use_synonyms": True,
            "automatically_extensible": True
        },
        "dummy_entity_2": {
            "data": [
                {
                    "value": "dummy_entity_2",
                    "synonyms": ["dummy_2"]
                }
            ],
            "use_synonyms": True,
            "automatically_extensible": True
        }
    },
    "language": "en"
}
DATASET = validate_and_format_dataset(DATASET)


# pylint: disable=unused-argument
def _persist_parser(self, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("nothing interesting here")


# pylint: disable=unused-argument
def _load_parser(self, path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()

# pylint: disable=unused-argument
def _stem(string, language):
    return string[:-1]

class TestCustomEntityParser(FixtureTest):

    @patch("snips_nlu_ontology.GazetteerEntityParser.persist")
    def test_should_persist_unfitted_parser(self, mocked_persist):
        # Given
        parser = CustomEntityParser(CustomEntityParserUsage.WITHOUT_STEMS)

        # When
        parser.persist(self.tmp_file_path)

        # Then
        expected_model = {
            "parser": None,
            "entities": None,
            "parser_usage": 0
        }
        expected_metadata = {"unit_name": "custom_entity_parser"}
        self.assertJsonContent(
            self.tmp_file_path / "custom_entities_parser.json", expected_model)
        self.assertJsonContent(
            self.tmp_file_path / "metadata.json", expected_metadata)
        mocked_persist.assert_not_called()

    @patch("snips_nlu_ontology.GazetteerEntityParser.persist")
    def test_should_persist_fitted_parser(self, mocked_persist):
        # Given
        mocked_persist.side_effect = _persist_parser
        parser = CustomEntityParser(CustomEntityParserUsage.WITHOUT_STEMS)

        # When
        parser.persist(self.tmp_file_path)

        # Then
        _parser_path = str(self.tmp_file_path / "parser")
        expected_model = {
            "parser": _parser_path,
            "entities": {"dummy_1", "dummy_2"},
            "parser_usage": 0
        }
        expected_metadata = {"unit_name": "custom_entity_parser"}
        self.assertJsonContent(
            self.tmp_file_path / "custom_entities_parser.json", expected_model)
        self.assertJsonContent(
            self.tmp_file_path / "metadata.json", expected_metadata)
        self.assertTrue(_parser_path.exists())
        with _parser_path.open("r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual("nothing interesting here", content.strip())

    @patch("snips_nlu_ontology.GazetteerEntityParser.load")
    def test_should_load_unfitted_parser(self, mocked_load):
        # Given
        _parser_path = str(self.tmp_file_path / "parser")
        parser_model = {
            "entities": None,
            "parser_usage": 0,
            "parser": None
        }
        metadata = {"unit_name": "custom_entity_parser"}

        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(
            self.tmp_file_path / "custom_entities_parser.json",
            parser_model
        )

        # When
        parser = CustomEntityParser.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(CustomEntityParserUsage.WITHOUT_STEMS,
                         parser.parser_usage)
        self.assertIsNone(parser.entities)
        self.assertIsNone(parser._parser)

        mocked_load.assert_not_called()

    @patch("snips_nlu_ontology.GazetteerEntityParser.load")
    def test_should_load_fitted_parser(self, mocked_load):
        # Given
        mocked_load.side_effect = _load_parser
        expected_entities = {"dummy_entity_1", "dummy_entity_2"}
        _parser_path = str(self.tmp_file_path / "parser")
        parser_model = {
            "entities": list(expected_entities),
            "parser_usage": 0,
            "parser": _parser_path
        }
        metadata = {"unit_name": "custom_entity_parser"}

        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(
            self.tmp_file_path / "custom_entities_parser.json",
            parser_model
        )

        with _parser_path.open("w", encoding="utf-8") as f:
            f.write("this is supposed to be as parser")

        # When
        parser = CustomEntityParser.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(expected_entities, parser.entities)
        self.assertEqual(
            CustomEntityParserUsage.WITHOUT_STEMS, parser.parser_usage)
        self.assertEqual("this is supposed to be as parser", parser._parser)

    def test_should_fit_and_parse(self):
        # Given
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)
        text = "dummy_entity_1 dummy_1 dummy_entity_2 dummy_2"

        # When
        result = parser.parse(text)

        # Then
        self.assertEqual(4, len(result))
        expected_entities = []  # TODO: fill the expected_entities
        for ent in result:
            self.assertIn(ent, expected_entities)

    def test_should_parse_with_stems(self):
        # Given
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITH_STEMS).fit(DATASET)
        text = "dummy_entity_1 dummy_1 dummy_entity_2 dummy_2"

        # When
        result = parser.parse(text)

        # Then
        self.assertEqual(4, len(result))
        expected_entities = []  # TODO: fill the expected_entities
        for ent in result:
            self.assertIn(ent, expected_entities)

    @patch("snips_nlu.entity_parser.custom_entity_parser.stem")
    def test_should_parse_with_and_without_stems(self, mocked_stem):
        # Given
        mocked_stem.side_effect = _stem
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)
        scope = ["dummy_entity_1"]
        text = "dummy_entity_ dummy_entity_1"

        # When
        result = parser.parse(text, scope=scope)

        # Then
        expected_entities = []  # TODO: fill the expected_entities
        self.assertEqual(2, len(result))
        self.assertEqual(expected_entities, result)


    def test_should_respect_scope(self):
        # Given
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)
        scope = ["dummy_entity_1"]
        text = "dummy_entity_2"

        # When
        result = parser.parse(text, scope=scope)

        # Then
        self.assertEqual(0, len(result))

    @patch("snips_nlu_ontology.GazetteerEntityParser.parse")
    def test_should_use_cache(self, mocked_parse):
        # Given
        mocked_parse.return_value = []
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)

        text = ""

        # When
        parser.parse(text)
        parser.parse(text)

        # Then
        self.assertEqual(2, mocked_parse.call_counts)
