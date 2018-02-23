from __future__ import unicode_literals

import io
import json
import os

from snips_nlu_ontology import get_all_languages

from snips_nlu import SnipsNLUEngine
from snips_nlu.constants import ROOT_PATH, LANGUAGE, RES_INTENT, \
    RES_INTENT_NAME
from snips_nlu.tests.utils import SnipsTest

SAMPLES_PATH = os.path.join(ROOT_PATH, "samples")


class TestSamples(SnipsTest):
    def setUp(self):
        sample_dataset_path = os.path.join(SAMPLES_PATH, "sample_dataset.json")
        with io.open(sample_dataset_path) as f:
            self.sample_dataset = json.load(f)

    def test_sample_configs_should_work(self):
        # Given
        dataset = self.sample_dataset

        for language in get_all_languages():
            # When
            config_file = "config_%s.json" % language
            config_path = os.path.join(SAMPLES_PATH, "configs", config_file)
            with io.open(config_path) as f:
                config = json.load(f)
            dataset[LANGUAGE] = language
            engine = SnipsNLUEngine(config).fit(dataset)
            result = engine.parse("Please give me the weather in Paris")

            # Then
            intent_name = result[RES_INTENT][RES_INTENT_NAME]
            self.assertEqual("sampleGetWeather", intent_name)
