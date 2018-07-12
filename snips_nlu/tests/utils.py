from __future__ import unicode_literals

import json
import shutil
import tempfile
import traceback as tb
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase

from builtins import bytes
from snips_nlu_ontology import get_all_languages

from snips_nlu.resources import load_resources

TEST_PATH = Path(__file__).parent
SAMPLE_DATASET_PATH = TEST_PATH / "resources" / "sample_dataset.json"
BEVERAGE_DATASET_PATH = TEST_PATH / "resources" / "beverage_dataset.json"
WEATHER_DATASET_PATH = TEST_PATH / "resources" / "weather_dataset.json"
PERFORMANCE_DATASET_PATH = TEST_PATH / "resources" / "performance_dataset.json"


class SnipsTest(TestCase):

    def __init__(self, methodName='runTest'):
        super(SnipsTest, self).__init__(methodName)
        for l in get_all_languages():
            load_resources(l)

    @contextmanager
    def fail_if_exception(self, msg):
        try:
            yield
        except Exception:  # pylint: disable=W0703
            trace = tb.format_exc()
            self.fail("{}\b{}".format(msg, trace))

    def assertJsonContent(self, json_path, expected_dict):
        if not json_path.exists():
            self.fail("Json file not found: %s" % str(json_path))
        with json_path.open() as f:
            data = json.load(f)
        self.assertDictEqual(expected_dict, data)

    def assertFileContent(self, path, expected_content):
        if not path.exists():
            self.fail("File not found: %s" % str(path))
        with path.open() as f:
            data = f.read()
        self.assertEqual(expected_content, data)

    @staticmethod
    def writeJsonContent(path, json_dict):
        json_content = bytes(json.dumps(json_dict), encoding="utf8")
        with path.open(mode="w") as f:
            f.write(json_content.decode("utf8"))

    @staticmethod
    def writeFileContent(path, content):
        content = bytes(content, encoding="utf8")
        with path.open(mode="w") as f:
            f.write(content.decode("utf8"))


class FixtureTest(SnipsTest):
    fixture_dir = TEST_PATH / "fixture"

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


def get_empty_dataset(language):
    return {
        "intents": {},
        "entities": {},
        "language": language,
        "snips_nlu_version": "1.1.1"
    }


with SAMPLE_DATASET_PATH.open(encoding='utf8') as dataset_file:
    SAMPLE_DATASET = json.load(dataset_file)

with BEVERAGE_DATASET_PATH.open(encoding='utf8') as dataset_file:
    BEVERAGE_DATASET = json.load(dataset_file)

with WEATHER_DATASET_PATH.open(encoding='utf8') as dataset_file:
    WEATHER_DATASET = json.load(dataset_file)

with PERFORMANCE_DATASET_PATH.open(encoding='utf8') as dataset_file:
    PERFORMANCE_DATASET = json.load(dataset_file)
