from __future__ import unicode_literals

import json
import traceback as tb
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase

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
