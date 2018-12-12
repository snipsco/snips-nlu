# coding=utf-8
import io
import shutil
import tempfile

from snips_nlu import SnipsNLUEngine
from snips_nlu.cli import (
    cross_val_metrics, parse, train, train_test_metrics, generate_dataset)
from snips_nlu.dataset import Dataset
from snips_nlu.tests.utils import SnipsTest, TEST_PATH, redirect_stdout
from snips_nlu.utils import unicode_string, json_string


def mk_sys_argv(args):
    return ["program_name"] + args


class TestCLI(SnipsTest):
    fixture_dir = TEST_PATH / "cli_fixture"

    # pylint: disable=protected-access
    def setUp(self):
        super(TestCLI, self).setUp()
        if not self.fixture_dir.exists():
            self.fixture_dir.mkdir()

        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups
- i want [number_of_cups] cups of [beverage_temperature](boiling hot) tea pls
- can you prepare [number_of_cups] cup of [beverage_temperature](cold) tea ?

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups] cups of coffee
- can you prepare [number_of_cups] cup of coffee""")
        beverage_dataset = Dataset.from_yaml_files("en", [dataset_stream]).json

        self.beverage_dataset_path = self.fixture_dir / "beverage_dataset.json"
        if self.beverage_dataset_path.exists():
            self.beverage_dataset_path.unlink()
        with self.beverage_dataset_path.open(mode="w") as f:
            f.write(json_string(beverage_dataset))

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
        train(self.beverage_dataset_path, str(self.tmp_file_path),
              config_path=None, verbose=False)

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No trained engine generated")
        msg = "Failed to create an engine from engine dict."
        with self.fail_if_exception(msg):
            SnipsNLUEngine.from_path(self.tmp_file_path)

    def test_parse(self):
        # Given / When
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
- brew [number_of_cups:snips/number](one) cup of coffee please
- make me [number_of_cups] cups of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        nlu_engine = SnipsNLUEngine().fit(dataset)
        nlu_engine.persist(self.tmp_file_path)

        # When / Then
        output_target = io.StringIO()
        with self.fail_if_exception("Failed to parse using CLI script"):
            with redirect_stdout(output_target):
                parse(str(self.tmp_file_path), "Make me two cups of coffee")
        output = output_target.getvalue()

        # Then
        expected_output = """{
  "input": "Make me two cups of coffee",
  "intent": {
    "intentName": "MakeCoffee",
    "probability": 1.0
  },
  "slots": [
    {
      "entity": "snips/number",
      "range": {
        "end": 11,
        "start": 8
      },
      "rawValue": "two",
      "slotName": "number_of_cups",
      "value": {
        "kind": "Number",
        "value": 2.0
      }
    }
  ]
}
"""
        self.assertEqual(expected_output, output)

    def test_generate_dataset(self):
        # Given
        yaml_string = """
# searchFlight Intent
---
type: intent
name: searchFlight
utterances:
  - find me a flight to [destination:city](Lima) [date:snips/datetime](tonight)

# City Entity
---
type: entity
name: city
values:
  - [new york, big apple]"""
        self.tmp_file_path = self.tmp_file_path.with_suffix(".yaml")
        with self.tmp_file_path.open(mode="w") as f:
            f.write(unicode_string(yaml_string))

        # When
        out = io.StringIO()
        with redirect_stdout(out):
            generate_dataset("en", str(self.tmp_file_path))
        printed_value = out.getvalue()

        # Then
        expected_value = """{
  "entities": {
    "city": {
      "automatically_extensible": true,
      "data": [
        {
          "synonyms": [
            "big apple"
          ],
          "value": "new york"
        }
      ],
      "matching_strictness": 1.0,
      "use_synonyms": true
    },
    "snips/datetime": {}
  },
  "intents": {
    "searchFlight": {
      "utterances": [
        {
          "data": [
            {
              "text": "find me a flight to "
            },
            {
              "entity": "city",
              "slot_name": "destination",
              "text": "Lima"
            },
            {
              "text": " "
            },
            {
              "entity": "snips/datetime",
              "slot_name": "date",
              "text": "tonight"
            }
          ]
        }
      ]
    }
  },
  "language": "en"
}
"""
        self.assertEqual(expected_value, printed_value)

    def test_cross_val_metrics(self):
        # Given / When
        cross_val_metrics(str(self.beverage_dataset_path),
                          str(self.tmp_file_path))

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No metrics found")

    def test_train_test_metrics(self):
        # Given / When
        train_test_metrics(
            str(self.beverage_dataset_path), str(self.beverage_dataset_path),
            str(self.tmp_file_path))

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No metrics found")
