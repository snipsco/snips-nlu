import io
import json
import os

from snips_nlu.dataset import validate_dataset

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                   "sample_dataset.json")

EMPTY_DATASET = {"intents": {}, "entities": {}}

with io.open(SAMPLE_DATASET_PATH) as dataset_file:
    SAMPLE_DATASET = json.load(dataset_file)

validate_dataset(EMPTY_DATASET)
validate_dataset(SAMPLE_DATASET)
