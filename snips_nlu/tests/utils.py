import io
import json
import os

from snips_nlu.dataset import validate_and_format_dataset

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATASET_PATH = os.path.join(TEST_PATH, "resources",
                                   "sample_dataset.json")

EMPTY_DATASET = {"intents": {}, "entities": {}, "language": u"en"}

with io.open(SAMPLE_DATASET_PATH) as dataset_file:
    SAMPLE_DATASET = json.load(dataset_file)

EMPTY_DATASET = validate_and_format_dataset(EMPTY_DATASET)
SAMPLE_DATASET = validate_and_format_dataset(SAMPLE_DATASET)
