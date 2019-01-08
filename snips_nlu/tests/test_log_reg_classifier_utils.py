# coding=utf-8
from __future__ import unicode_literals

import io
from copy import deepcopy
from itertools import cycle

import numpy as np
from numpy.random.mtrand import RandomState

from future.utils import itervalues
from mock import MagicMock, patch

from snips_nlu.constants import INTENTS, LANGUAGE_EN, UTTERANCES
from snips_nlu.dataset import validate_and_format_dataset, Dataset
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    add_unknown_word_to_utterances, build_training_data,
    generate_noise_utterances, generate_smart_noise, get_noise_it,
    remove_builtin_slots, text_to_utterance, get_dataset_specific_noise)
from snips_nlu.pipeline.configs import (
    IntentClassifierDataAugmentationConfig, LogRegIntentClassifierConfig)
from snips_nlu.tests.test_log_reg_intent_classifier import (
    get_mocked_augment_utterances)
from snips_nlu.tests.utils import SnipsTest, get_empty_dataset


class TestLogRegClassifierUtils(SnipsTest):
    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils"
           ".augment_utterances")
    def test_should_build_training_data_with_no_noise(
            self, mocked_augment_utterances):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: my_first_intent
utterances:
- how are you
- hello how are you?
- what's up

---
type: intent
name: my_second_intent
utterances:
- what is the weather today ?
- does it rain
- will it rain tomorrow""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances
        random_state = np.random.RandomState(1)

        # When
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            noise_factor=0)
        utterances, _, intent_mapping = build_training_data(
            dataset, LANGUAGE_EN, data_augmentation_config, random_state)

        # Then
        expected_utterances = [utterance for intent
                               in itervalues(dataset[INTENTS])
                               for utterance in intent[UTTERANCES]]
        expected_intent_mapping = ["my_first_intent", "my_second_intent"]
        self.assertListEqual(expected_utterances, utterances)
        self.assertListEqual(expected_intent_mapping, intent_mapping)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noise")
    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils"
           ".augment_utterances")
    def test_should_build_training_data_with_noise(
            self, mocked_augment_utterances, mocked_get_noise):
        # Given
        mocked_noises = ["mocked_noise_%s" % i for i in range(100)]
        mocked_get_noise.return_value = mocked_noises
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances

        num_intents = 3
        utterances_length = 5
        num_queries_per_intent = 3
        fake_utterance = {
            "data": [
                {"text": " ".join("1" for _ in range(utterances_length))}
            ]
        }
        dataset = {
            "intents": {
                str(i): {
                    "utterances": [fake_utterance] * num_queries_per_intent
                } for i in range(num_intents)
            },
            "entities": {}
        }
        random_state = np.random.RandomState(1)

        # When
        np.random.seed(42)
        noise_factor = 2
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            noise_factor=noise_factor, unknown_word_prob=0,
            unknown_words_replacement_string=None)
        utterances, _, intent_mapping = build_training_data(
            dataset, LANGUAGE_EN, data_augmentation_config, random_state)

        # Then
        expected_utterances = [utterance
                               for intent in itervalues(dataset[INTENTS])
                               for utterance in intent[UTTERANCES]]
        np.random.seed(42)
        noise = list(mocked_noises)
        noise_size = int(min(noise_factor * num_queries_per_intent,
                             len(noise)))
        noise_it = get_noise_it(mocked_noises, utterances_length, 0,
                                random_state)
        noisy_utterances = [text_to_utterance(next(noise_it))
                            for _ in range(noise_size)]
        expected_utterances += noisy_utterances
        expected_intent_mapping = sorted(dataset["intents"])
        expected_intent_mapping.append(None)
        self.assertListEqual(expected_utterances, utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    def test_add_unknown_words_to_utterances(self):
        # Given
        base_utterances = {
            "data": [
                {
                    "text": "hello "
                },
                {
                    "text": " you ",
                    "entity": "you"
                },
                {
                    "text": " how are you "
                },
                {
                    "text": "cat",
                    "entity": "cat"
                }
            ]
        }
        utterances = []
        for _ in range(6):
            utterances.append(deepcopy(base_utterances))

        rand_it = cycle([0, 1])

        def mocked_rand():
            return next(rand_it)

        max_unknown_words = 3
        rg_it = cycle([i for i in range(1, max_unknown_words + 1)])

        def mocked_randint(a, b):  # pylint: disable=unused-argument
            return next(rg_it)

        unknownword_prob = .5

        random_state = MagicMock()
        random_state_rand = MagicMock()
        random_state_rand.side_effect = mocked_rand
        random_state_choice = MagicMock()
        random_state_choice.side_effect = mocked_randint

        random_state.rand = random_state_rand
        random_state.randint = random_state_choice

        # When
        replacement_string = "unknownword"
        noisy_utterances = add_unknown_word_to_utterances(
            utterances, unknown_word_prob=unknownword_prob,
            replacement_string=replacement_string,
            max_unknown_words=max_unknown_words,
            random_state=random_state
        )

        # Then
        expected_utterances = [
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "cat",
                        "entity": "cat"
                    },
                    {
                        "text": " unknownword"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "cat",
                        "entity": "cat"
                    },
                ]
            },
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "cat",
                        "entity": "cat"
                    },
                    {
                        "text": " unknownword unknownword"
                    }
                ]
            },
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "cat",
                        "entity": "cat"
                    },
                ]
            },
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "cat",
                        "entity": "cat"
                    },
                    {
                        "text": " unknownword unknownword unknownword"
                    }

                ]
            },
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "cat",
                        "entity": "cat"
                    },
                ]
            }
        ]
        self.assertEqual(expected_utterances, noisy_utterances)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noise")
    def test_generate_noise_utterances_should_replace_unknown_words(
            self, mocked_noise):
        # Given
        utterances = [
            {
                "data": [
                    {
                        "text": "hello "
                    },
                    {
                        "text": " you ",
                        "entity": "you"
                    },
                    {
                        "text": " how are you "
                    },
                    {
                        "text": "bobby",
                        "entity": "you"
                    }
                ]
            }
        ]
        language = LANGUAGE_EN
        base_noise = ["hello", "dear", "you", "fool"]
        mocked_noise.return_value = base_noise
        replacement_string = "unknownword"

        # When
        noise = generate_smart_noise(
            base_noise, utterances, replacement_string, language)

        # Then
        expected_noise = ["hello", replacement_string, "you",
                          replacement_string]
        self.assertEqual(noise, expected_noise)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noise")
    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils"
           ".augment_utterances")
    def test_should_build_training_data_with_unknown_noise(
            self, mocked_augment_utterances, mocked_get_noise):
        # Given
        mocked_noises = ["mocked_noise_%s" % i for i in range(100)]
        mocked_get_noise.return_value = mocked_noises
        mocked_augment_utterances.side_effect = get_mocked_augment_utterances

        num_intents = 3
        utterances_length = 5
        num_queries_per_intent = 3
        fake_utterance = {
            "data": [
                {"text": " ".join("1" for _ in range(utterances_length))}
            ]
        }
        dataset = {
            "intents": {
                str(i): {
                    "utterances": [fake_utterance] * num_queries_per_intent
                } for i in range(num_intents)
            },
            "entities": {}
        }
        random_state = np.random.RandomState(1)

        # When
        np.random.seed(42)
        noise_factor = 2
        replacement_string = "unknownword"
        data_augmentation_config = IntentClassifierDataAugmentationConfig(
            noise_factor=noise_factor, unknown_word_prob=0,
            unknown_words_replacement_string=replacement_string)
        utterances, _, intent_mapping = build_training_data(
            dataset, LANGUAGE_EN, data_augmentation_config, random_state)

        # Then
        expected_utterances = [utterance
                               for intent in itervalues(dataset[INTENTS])
                               for utterance in intent[UTTERANCES]]
        np.random.seed(42)
        noise = list(mocked_noises)
        noise_size = int(min(noise_factor * num_queries_per_intent,
                             len(noise)))
        noisy_utterances = [text_to_utterance(replacement_string)
                            for _ in range(noise_size)]
        expected_utterances += noisy_utterances
        expected_intent_mapping = sorted(dataset["intents"])
        expected_intent_mapping.append(None)
        self.assertListEqual(expected_utterances, utterances)
        self.assertListEqual(expected_intent_mapping, intent_mapping)

    def test_should_build_training_data_with_no_data(self):
        # Given
        language = LANGUAGE_EN
        dataset = validate_and_format_dataset(get_empty_dataset(language))
        random_state = np.random.RandomState(1)

        # When
        data_augmentation_config = LogRegIntentClassifierConfig() \
            .data_augmentation_config
        utterances, _, intent_mapping = build_training_data(
            dataset, language, data_augmentation_config, random_state)

        # Then
        expected_utterances = []
        expected_intent_mapping = []
        self.assertListEqual(utterances, expected_utterances)
        self.assertListEqual(intent_mapping, expected_intent_mapping)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noise")
    def test_generate_noise_utterances(self, mocked_get_noise):
        # Given
        language = LANGUAGE_EN
        num_intents = 2
        noise_factor = 1
        utterances_length = 5

        noise = [str(i) for i in range(utterances_length)]
        mocked_get_noise.return_value = noise

        augmented_utterances = [
            {
                "data": [
                    {
                        "text": " ".join(
                            "{}".format(i) for i in range(utterances_length))
                    }
                ]
            }
        ]
        num_utterances = 10
        random_state = np.random.RandomState(1)

        augmented_utterances = augmented_utterances * num_utterances
        config = IntentClassifierDataAugmentationConfig(
            noise_factor=noise_factor)
        # When
        noise_utterances = generate_noise_utterances(
            augmented_utterances, noise, num_intents, config, language,
            random_state)

        # Then
        joined_noise = text_to_utterance(" ".join(noise))
        for u in noise_utterances:
            self.assertEqual(u, joined_noise)

    def test_remove_builtin_slots(self):
        # Given
        language = LANGUAGE_EN
        dataset = {
            "entities": {
                "snips/number": {}
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "I want ",
                                },
                                {
                                    "text": "three",
                                    "slot_name": "number_of_cups",
                                    "entity": "snips/number"
                                },
                                {
                                    "text": " cups",
                                },
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "give me ",
                                },
                                {
                                    "text": "twenty two",
                                    "slot_name": "number_of_cups",
                                    "entity": "snips/number"
                                },
                                {
                                    "text": " big cups please",
                                },
                            ]
                        }
                    ]
                }
            },
            "language": language
        }

        # When
        filtered_dataset = remove_builtin_slots(dataset)

        # Then
        expected_dataset = {
            "entities": {
                "snips/number": {}
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "I want ",
                                },
                                {
                                    "text": " cups",
                                },
                            ]
                        },
                        {
                            "data": [
                                {
                                    "text": "give me ",
                                },
                                {
                                    "text": " big cups please",
                                },
                            ]
                        }
                    ]
                }
            },
            "language": language
        }

        self.assertDictEqual(expected_dataset, filtered_dataset)

    @patch("snips_nlu.intent_classifier.log_reg_classifier_utils.get_noise")
    def test_get_dataset_specific_noise(self, mocked_noise):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: my_intent
utterances:
- what is the weather in [city](paris)
- give me the weather in [city](london) 
- does it rain in [city](tokyo)?""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        dataset = validate_and_format_dataset(dataset)
        language = "en"
        mocked_noise.return_value = ["paris", "tokyo", "yo"]
        # When
        noise = get_dataset_specific_noise(dataset, language)

        # Then
        self.assertEqual(["yo"], noise)

    def test_add_unknown_word_to_utterances_with_none_max_unknownword(self):
        # Given
        utterances = [text_to_utterance("yo")]
        replacement_string = "yo"
        unknown_word_prob = 1
        max_unknown_words = None
        random_state = RandomState()

        # When / Then
        with self.fail_if_exception(
            "Failed to augment utterances with max_unknownword=None"):
            add_unknown_word_to_utterances(
                utterances, replacement_string, unknown_word_prob,
                max_unknown_words, random_state
            )

    def test_add_unknown_word_to_utterances_with_zero_max_unknownword(self):
        # Given
        utterances = [text_to_utterance("yo")]
        replacement_string = "yo"
        unknown_word_prob = 1
        max_unknown_words = 0
        random_state = RandomState()

        # When / Then
        with self.fail_if_exception(
            "Failed to augment utterances with unknown_word_prob=0"):
            add_unknown_word_to_utterances(
                utterances, replacement_string, unknown_word_prob,
                max_unknown_words, random_state
            )
