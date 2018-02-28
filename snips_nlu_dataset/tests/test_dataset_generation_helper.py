import os
import unittest

from snips_nlu.constants import ROOT_PATH
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu_dataset import AssistantDataset


class TestDatasetGenerationHelper(unittest.TestCase):
    def test_should_generate_dataset_from_file(self):
        # Given
        dataset_path_1 = os.path.join(ROOT_PATH, "snips_nlu_dataset",
                                      "examples", "whoIsGame.txt")
        dataset_path_2 = os.path.join(ROOT_PATH, "snips_nlu_dataset",
                                      "examples", "getWeather.txt")
        dataset = AssistantDataset.from_files(
            "en", [dataset_path_1, dataset_path_2])
        dataset_dict = dataset.json

        # When / Then
        validate_and_format_dataset(dataset_dict)
