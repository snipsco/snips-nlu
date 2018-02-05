from __future__ import unicode_literals

import io
import json
import os

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                   "sample_dataset.json")
BEVERAGE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                     "beverage_dataset.json")

PERFORMANCE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                        "performance_dataset.json")


def get_empty_dataset(language):
    return {
        "intents": {},
        "entities": {},
        "language": language.iso_code,
        "snips_nlu_version": "1.1.1"
    }


with io.open(SAMPLE_DATASET_PATH, encoding='utf8') as dataset_file:
    SAMPLE_DATASET = json.load(dataset_file)

with io.open(BEVERAGE_DATASET_PATH, encoding='utf8') as dataset_file:
    BEVERAGE_DATASET = json.load(dataset_file)

with io.open(PERFORMANCE_DATASET_PATH, encoding='utf8') as dataset_file:
    PERFORMANCE_DATASET = json.load(dataset_file)
