import unittest

import numpy as np
from mock import patch

from snips_nlu.constants import INTENTS, UTTERANCES, DATA
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.intent_classifier.data_augmentation import build_training_data
from snips_nlu.languages import Language
from snips_nlu.tests.utils import SAMPLE_DATASET, empty_dataset


class TestDataAugmentation(unittest.TestCase):
    def test_should_build_training_data_with_no_stemming_no_noise(self):
        # Given
        dataset = SAMPLE_DATASET

        # When
        utterances, y, intent_mapping = build_training_data(dataset,
                                                            Language.EN,
                                                            use_stemming=False,
                                                            noise_factor=0)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA]) for intent
                               in dataset[INTENTS].values() for utterance in
                               intent[UTTERANCES]]
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1']
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    @patch("snips_nlu.intent_classifier.data_augmentation.get_subtitles")
    def test_should_build_training_data_with_noise(self, mocked_get_subtitles):
        # Given
        mocked_subtitles = set("mocked_subtitle_%s" % i for i in xrange(100))
        mocked_get_subtitles.return_value = mocked_subtitles

        dataset = SAMPLE_DATASET
        nb_utterances = [len(intent[UTTERANCES]) for intent in
                         dataset[INTENTS].values()]
        avg_utterances = np.mean(nb_utterances)

        # When
        # noinspection PyUnresolvedReferences
        np.random.seed(42)
        noise_factor = 2
        utterances, y, intent_mapping = build_training_data(
            dataset, Language.EN, use_stemming=False,
            noise_factor=noise_factor)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA])
                               for intent in dataset[INTENTS].values()
                               for utterance in intent[UTTERANCES]]
        # noinspection PyUnresolvedReferences
        np.random.seed(42)
        noise = list(mocked_subtitles)
        # noinspection PyTypeChecker
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

        def get_mocked_stem(string, _):
            return "[STEMMED] %s" % string

        mocked_stem.side_effect = get_mocked_stem

        # When
        utterances, y, intent_mapping = build_training_data(
            dataset, Language.EN, use_stemming=True, noise_factor=0)

        # Then
        expected_utterances = [get_text_from_chunks(utterance[DATA])
                               for intent in dataset[INTENTS].values()
                               for utterance in intent[UTTERANCES]]
        expected_utterances = ["[STEMMED] %s" % utterance for utterance in
                               expected_utterances]
        expected_intent_mapping = [u'dummy_intent_2', u'dummy_intent_1']
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    def test_should_build_training_data_with_no_data(self):
        # Given
        language = Language.EN
        dataset = empty_dataset(language)

        # When
        utterances, y, intent_mapping = build_training_data(
            dataset, language, use_stemming=False, noise_factor=0)

        # Then
        expected_utterances = []
        expected_intent_mapping = []
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)
