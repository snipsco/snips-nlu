import os
import unittest

from nlu_dataset import AssistantDataset
from snips_nlu.constants import ROOT_PATH
from snips_nlu.dataset import validate_and_format_dataset


class TestDatasetGenerationHelper(unittest.TestCase):
    def test_should_generate_dataset_from_file(self):
        # Given
        dataset_path_1 = os.path.join(ROOT_PATH, "nlu_dataset", "examples",
                                      "who_is_game.txt")
        dataset_path_2 = os.path.join(ROOT_PATH, "nlu_dataset", "examples",
                                      "get_weather.txt")
        dataset = AssistantDataset.from_files(
            "en", [dataset_path_1, dataset_path_2])
        dataset_dict = dataset.json

        # When / Then
        validate_and_format_dataset(dataset_dict)
