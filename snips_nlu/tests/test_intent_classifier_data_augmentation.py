import unittest

import numpy as np
from copy import deepcopy
from mock import patch

from snips_nlu.constants import (CUSTOM_ENGINE, BUILTIN_ENGINE, INTENTS,
                                 UTTERANCES, UTTERANCE_TEXT, ENGINE_TYPE, DATA,
                                 TEXT)
from snips_nlu.dataset import filter_dataset, validate_and_format_dataset
from snips_nlu.intent_classifier.data_augmentation import build_training_data
from snips_nlu.languages import Language
from snips_nlu.tests.utils import SAMPLE_DATASET, empty_dataset


class TestDataAugmentation(unittest.TestCase):
    def test_should_build_training_data_with_no_stemming_no_noise(self):
        # Given
        dataset = SAMPLE_DATASET
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE,
                                        min_utterances=1)
        builtin_dataset = filter_dataset(dataset, BUILTIN_ENGINE,
                                         min_utterances=1)

        # When
        utterances, y, intent_mapping = build_training_data(custom_dataset,
                                                            builtin_dataset,
                                                            Language.EN,
                                                            use_stemming=False,
                                                            noise_factor=0)

        # Then
        expected_utterances = [utterance[UTTERANCE_TEXT] for intent in
                               dataset[INTENTS].values() for utterance in
                               intent[UTTERANCES]]
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1']
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    @patch("snips_nlu.intent_classifier.data_augmentation.get_subtitles")
    def test_should_build_training_data_with_noise(self, mocked_get_subtitles):
        # Given
        def get_mocked_subtitles(_):
            return set("mocked_subtitle_%s" % i for i in xrange(100))

        mocked_get_subtitles.side_effect = get_mocked_subtitles

        dataset = SAMPLE_DATASET
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE,
                                        min_utterances=1)
        builtin_dataset = filter_dataset(dataset, BUILTIN_ENGINE,
                                         min_utterances=1)
        nb_utterances = [len(intent[UTTERANCES]) for intent in
                         custom_dataset[INTENTS].values()]
        avg_utterances = np.mean(nb_utterances)

        # When
        np.random.seed(42)
        noise_factor = 2
        utterances, y, intent_mapping = build_training_data(custom_dataset,
                                                            builtin_dataset,
                                                            Language.EN,
                                                            use_stemming=False,
                                                            noise_factor=noise_factor)

        # Then
        expected_utterances = [utterance[UTTERANCE_TEXT] for intent in
                               dataset[INTENTS].values() for utterance in
                               intent[UTTERANCES]]
        np.random.seed(42)
        noise = list(get_mocked_subtitles(Language.EN))
        noise_size = int(min(noise_factor * avg_utterances, len(noise)))
        noisy_utterances = np.random.choice(noise, size=noise_size,
                                            replace=False)
        expected_utterances += list(noisy_utterances)
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1', None]
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    @patch("snips_nlu.intent_classifier.data_augmentation.stem_sentence")
    def test_should_build_training_data_with_stemming(self, mocked_stem):
        # Given
        dataset = SAMPLE_DATASET
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE,
                                        min_utterances=1)
        builtin_dataset = filter_dataset(dataset, BUILTIN_ENGINE,
                                         min_utterances=1)

        def get_mocked_stem(string, _):
            return "[STEMMED] %s" % string

        mocked_stem.side_effect = get_mocked_stem

        # When
        utterances, y, intent_mapping = build_training_data(custom_dataset,
                                                            builtin_dataset,
                                                            Language.EN,
                                                            use_stemming=True,
                                                            noise_factor=0)

        # Then
        expected_utterances = [utterance[UTTERANCE_TEXT] for intent in
                               dataset[INTENTS].values() for utterance in
                               intent[UTTERANCES]]
        expected_utterances = ["[STEMMED] %s" % utterance for utterance in
                               expected_utterances]
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1']
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    def test_should_build_training_data_with_builtin_data(self):
        # Given
        builtin_intent_1_utterances = [
            {
                DATA: [{
                    TEXT: u"this is the builtin_intent_1 utterance number %s"
                          % i
                }]
            }
            for i in xrange(50)
        ]

        builtin_intent_2_utterances = [
            {
                DATA: [{
                    TEXT: u"this is the builtin_intent_2 utterance number %s"
                          % i
                }]
            }
            for i in xrange(50)
        ]

        builtin_intents = {
            "builtin_intent_1": {
                ENGINE_TYPE: BUILTIN_ENGINE,
                UTTERANCES: builtin_intent_1_utterances
            },
            "builtin_intent_2": {
                ENGINE_TYPE: BUILTIN_ENGINE,
                UTTERANCES: builtin_intent_2_utterances
            }
        }

        dataset = deepcopy(SAMPLE_DATASET)
        dataset[INTENTS].update(builtin_intents)
        dataset = validate_and_format_dataset(dataset)
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE,
                                        min_utterances=1)
        builtin_dataset = filter_dataset(dataset, BUILTIN_ENGINE,
                                         min_utterances=1)

        # When
        utterances, y, intent_mapping = build_training_data(custom_dataset,
                                                            builtin_dataset,
                                                            Language.EN,
                                                            use_stemming=False,
                                                            noise_factor=0)

        # Then
        max_utterances = max(len(intent[UTTERANCES]) for intent in
                             custom_dataset[INTENTS].values())
        expected_utterances = [utterance[UTTERANCE_TEXT] for intent in
                               custom_dataset[INTENTS].values() for utterance
                               in intent[UTTERANCES]]
        expected_utterances += [utterance[UTTERANCE_TEXT] for intent in
                                builtin_dataset[INTENTS].values() for utterance
                                in intent[UTTERANCES][:max_utterances]]
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1', None]
        self.assertItemsEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    def test_should_build_training_data_without_customs(self):
        # Given
        language = Language.EN
        builtin_intent_1_utterances = [
            {
                DATA: [{
                    TEXT: u"this is the builtin_intent_1 utterance number %s"
                          % i
                }]
            }
            for i in xrange(50)
        ]

        builtin_intent_2_utterances = [
            {
                DATA: [{
                    TEXT: u"this is the builtin_intent_2 utterance number %s"
                          % i
                }]
            }
            for i in xrange(50)
        ]

        builtin_intents = {
            "builtin_intent_1": {
                ENGINE_TYPE: BUILTIN_ENGINE,
                UTTERANCES: builtin_intent_1_utterances
            },
            "builtin_intent_2": {
                ENGINE_TYPE: BUILTIN_ENGINE,
                UTTERANCES: builtin_intent_2_utterances
            }
        }

        dataset = empty_dataset(language)
        dataset[INTENTS].update(builtin_intents)
        dataset = validate_and_format_dataset(dataset)
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE,
                                        min_utterances=1)
        builtin_dataset = filter_dataset(dataset, BUILTIN_ENGINE,
                                         min_utterances=1)

        # When
        utterances, y, intent_mapping = build_training_data(custom_dataset,
                                                            builtin_dataset,
                                                            language,
                                                            use_stemming=False,
                                                            noise_factor=0)

        # Then
        expected_utterances = []
        expected_intent_mapping = [None]
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    def test_should_build_training_data_with_no_data(self):
        # Given
        language = Language.EN
        dataset = empty_dataset(language)
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE,
                                        min_utterances=1)
        builtin_dataset = filter_dataset(dataset, BUILTIN_ENGINE,
                                         min_utterances=1)

        # When
        utterances, y, intent_mapping = build_training_data(custom_dataset,
                                                            builtin_dataset,
                                                            language,
                                                            use_stemming=False,
                                                            noise_factor=0)

        # Then
        expected_utterances = []
        expected_intent_mapping = []
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)
