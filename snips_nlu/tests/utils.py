from __future__ import unicode_literals

import io
import json
import os
import traceback as tb
from contextlib import contextmanager
from unittest import TestCase

from snips_nlu_ontology import get_all_languages

from snips_nlu.resources import load_resources

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                   "sample_dataset.json")
BEVERAGE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                     "beverage_dataset.json")
WEATHER_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                    "weather_dataset.json")

PERFORMANCE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                        "performance_dataset.json")


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
            trace = tb.format_exc
            self.fail("{}\b{}".format(msg, trace))


def get_empty_dataset(language):
    return {
        "intents": {},
        "entities": {},
        "language": language,
        "snips_nlu_version": "1.1.1"
    }


with io.open(SAMPLE_DATASET_PATH, encoding='utf8') as dataset_file:
    SAMPLE_DATASET = json.load(dataset_file)

with io.open(BEVERAGE_DATASET_PATH, encoding='utf8') as dataset_file:
    BEVERAGE_DATASET = json.load(dataset_file)

with io.open(WEATHER_DATASET_PATH, encoding='utf8') as dataset_file:
    WEATHER_DATASET = json.load(dataset_file)

with io.open(PERFORMANCE_DATASET_PATH, encoding='utf8') as dataset_file:
    PERFORMANCE_DATASET = json.load(dataset_file)
