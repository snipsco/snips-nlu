from __future__ import unicode_literals

import io
import json
import os
import shutil
import unittest

from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.tests.utils import TEST_PATH
from snips_nlu.tokenization import tokenize


class TestCRFTagger(unittest.TestCase):
    def setUp(self):
        fixtures_directory = os.path.join(TEST_PATH, "fixtures", "crf_tagger")
        self.expected_tagger_directory = os.path.join(fixtures_directory,
                                                      "expected_output")
        self.actual_tagger_directory = os.path.join(fixtures_directory,
                                                    "actual_output")

    def tearDown(self):
        if os.path.isdir(self.actual_tagger_directory):
            shutil.rmtree(self.actual_tagger_directory)

    def test_should_be_saveable(self):
        # Given
        crf_model_filename = os.path.join(self.actual_tagger_directory,
                                          "model.crfsuite")
        crf_model = default_crf_model(model_filename=crf_model_filename)

        features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2},
                "offsets": [-1, 0]
            }
        ]
        tagging_scheme = TaggingScheme.BILOU
        data = [
            {
                "tokens": tokenize("I love blue birds"),
                "tags": ["O", "O", "B-COLOR", "O"]
            },
            {
                "tokens": tokenize("I like red birds"),
                "tags": ["O", "O", "B-COLOR", "O"]
            }
        ]

        tagger = CRFTagger(crf_model, features_signatures, tagging_scheme,
                           Language.EN)
        tagger.fit(data)

        # When
        tagger.save(self.actual_tagger_directory)

        # Then
        with io.open(os.path.join(self.expected_tagger_directory,
                                  "tagger_config.json")) as f:
            expected_config = json.load(f)

        with io.open(os.path.join(self.actual_tagger_directory,
                                  "tagger_config.json")) as f:
            actual_config = json.load(f)

        self.assertTrue(os.path.exists(crf_model_filename))
        self.assertDictEqual(actual_config, expected_config)

    def test_should_be_loadable(self):
        # When
        tagger = CRFTagger.load(self.expected_tagger_directory)

        # Then
        expected_features_signatures = [
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 1},
                "offsets": [0]
            },
            {
                "factory_name": "get_shape_ngram_fn",
                "args": {"n": 2},
                "offsets": [-1, 0]
            }
        ]
        expected_tagging_scheme = TaggingScheme.BILOU
        expected_crf_model_filename = os.path.join(
            self.expected_tagger_directory, "model.crfsuite")
        expected_language = Language.EN

        self.assertListEqual(tagger.features_signatures,
                             expected_features_signatures)
        self.assertEqual(tagger.tagging_scheme, expected_tagging_scheme)
        self.assertEqual(tagger.crf_model.modelfile.name,
                         expected_crf_model_filename)
        self.assertEqual(tagger.language, expected_language)
