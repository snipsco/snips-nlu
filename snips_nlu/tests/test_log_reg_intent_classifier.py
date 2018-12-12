# coding=utf-8
from __future__ import unicode_literals

import io
from builtins import str

from mock import patch

from snips_nlu.constants import (
    INTENTS, LANGUAGE_EN, RES_INTENT_NAME, RES_PROBA, UTTERANCES)
from snips_nlu.dataset import Dataset
from snips_nlu.entity_parser import BuiltinEntityParser, CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser_usage import (
    CustomEntityParserUsage)
from snips_nlu.exceptions import NotTrained
from snips_nlu.intent_classifier import LogRegIntentClassifier
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    text_to_utterance)
from snips_nlu.pipeline.configs import LogRegIntentClassifierConfig
from snips_nlu.tests.utils import FixtureTest, get_empty_dataset


# pylint: disable=unused-argument
def get_mocked_augment_utterances(dataset, intent_name, language,
                                  min_utterances, capitalization_ratio,
                                  add_builtin_entities_examples,
                                  random_state):
    return dataset[INTENTS][intent_name][UTTERANCES]


# pylint: enable=unused-argument


class TestLogRegIntentClassifier(FixtureTest):
    def test_should_get_intent(self):
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
        config = LogRegIntentClassifierConfig(random_seed=42)
        classifier = LogRegIntentClassifier(config).fit(dataset)
        text = "hey how are you doing ?"

        # When
        res = classifier.get_intent(text)
        intent = res[RES_INTENT_NAME]

        # Then
        self.assertEqual("my_first_intent", intent)

    def test_should_get_none_intent_when_empty_input(self):
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
        classifier = LogRegIntentClassifier().fit(dataset)
        text = ""

        # When
        result = classifier.get_intent(text)

        # Then
        self.assertIsNone(result)

    def test_should_get_intent_when_filter(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a cup of tea
- i want two cups of tea please
- can you prepare one cup of tea ?

---
type: intent
name: MakeCoffee
utterances:
- make me a cup of coffee please
- brew two cups of coffee
- can you prepare one cup of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        config = LogRegIntentClassifierConfig(random_seed=42)
        classifier = LogRegIntentClassifier(config).fit(dataset)

        # When
        text1 = "Make me two cups of tea"
        res1 = classifier.get_intent(text1, ["MakeCoffee", "MakeTea"])

        text2 = "Make me two cups of tea"
        res2 = classifier.get_intent(text2, ["MakeCoffee"])

        text3 = "bla bla bla"
        res3 = classifier.get_intent(text3, ["MakeCoffee"])

        # Then
        self.assertEqual("MakeTea", res1[RES_INTENT_NAME])
        self.assertEqual("MakeCoffee", res2[RES_INTENT_NAME])
        self.assertEqual(None, res3)

    def test_should_raise_when_not_fitted(self):
        # Given
        intent_classifier = LogRegIntentClassifier()

        # When / Then
        self.assertFalse(intent_classifier.fitted)
        with self.assertRaises(NotTrained):
            intent_classifier.get_intent("foobar")

    def test_should_get_none_intent_when_empty_dataset(self):
        # Given
        dataset = get_empty_dataset(LANGUAGE_EN)
        classifier = LogRegIntentClassifier().fit(dataset)
        text = "this is a dummy query"

        # When
        intent = classifier.get_intent(text)

        # Then
        expected_intent = None
        self.assertEqual(intent, expected_intent)

    def test_should_get_intents(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - yala yili

---
type: intent
name: intent2
utterances:
  - yala yili yulu

---
type: intent
name: intent3
utterances:
  - yili yulu yele""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        config = LogRegIntentClassifierConfig(random_seed=42)
        classifier = LogRegIntentClassifier(config).fit(dataset)
        text = "yala yili yulu"

        # When
        results = classifier.get_intents(text)
        intents = [res[RES_INTENT_NAME] for res in results]

        # Then
        expected_intents = ["intent2", "intent1", "intent3", None]

        self.assertEqual(expected_intents, intents)

    def test_should_get_intents_when_empty_dataset(self):
        # Given
        dataset = get_empty_dataset(LANGUAGE_EN)
        classifier = LogRegIntentClassifier().fit(dataset)
        text = "this is a dummy query"

        # When
        results = classifier.get_intents(text)

        # Then
        expected_results = [{RES_INTENT_NAME: None, RES_PROBA: 1.0}]
        self.assertEqual(expected_results, results)

    def test_should_get_intents_when_empty_input(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - foo bar

---
type: intent
name: intent2
utterances:
  - lorem ipsum""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        classifier = LogRegIntentClassifier().fit(dataset)
        text = ""

        # When
        results = classifier.get_intents(text)

        # Then
        expected_results = [
            {RES_INTENT_NAME: None, RES_PROBA: 1.0},
            {RES_INTENT_NAME: "intent1", RES_PROBA: 0.0},
            {RES_INTENT_NAME: "intent2", RES_PROBA: 0.0},
        ]
        self.assertEqual(expected_results, results)

    @patch('snips_nlu.intent_classifier.featurizer.Featurizer.to_dict')
    def test_should_be_serializable(self, mock_to_dict):
        # Given
        mocked_dict = {"mocked_featurizer_key": "mocked_featurizer_value"}
        mock_to_dict.return_value = mocked_dict

        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - foo bar

---
type: intent
name: intent2
utterances:
  - lorem ipsum""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        intent_classifier = LogRegIntentClassifier().fit(dataset)
        coeffs = intent_classifier.classifier.coef_.tolist()
        intercept = intent_classifier.classifier.intercept_.tolist()

        # When
        intent_classifier.persist(self.tmp_file_path)

        # Then
        intent_list = ["intent1", "intent2", None]
        expected_dict = {
            "config": LogRegIntentClassifierConfig().to_dict(),
            "coeffs": coeffs,
            "intercept": intercept,
            "t_": 701.0,
            "intent_list": intent_list,
            "featurizer": mocked_dict
        }
        metadata = {"unit_name": "log_reg_intent_classifier"}
        self.assertJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.assertJsonContent(self.tmp_file_path / "intent_classifier.json",
                               expected_dict)

    @patch('snips_nlu.intent_classifier.featurizer.Featurizer.from_dict')
    def test_should_be_deserializable(self, mock_from_dict):
        # Given
        mocked_featurizer = Featurizer(LANGUAGE_EN, None)
        mock_from_dict.return_value = mocked_featurizer

        intent_list = ["MakeCoffee", "MakeTea", None]

        coeffs = [
            [1.23, 4.5],
            [6.7, 8.90],
            [1.01, 2.345],
        ]

        intercept = [
            0.34,
            0.41,
            -0.98
        ]

        t_ = 701.

        config = LogRegIntentClassifierConfig().to_dict()

        classifier_dict = {
            "coeffs": coeffs,
            "intercept": intercept,
            "t_": t_,
            "intent_list": intent_list,
            "config": config,
            "featurizer": mocked_featurizer.to_dict(),
        }
        self.tmp_file_path.mkdir()
        metadata = {"unit_name": "log_reg_intent_classifier"}
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(self.tmp_file_path / "intent_classifier.json",
                              classifier_dict)

        # When
        classifier = LogRegIntentClassifier.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(classifier.intent_list, intent_list)
        self.assertIsNotNone(classifier.featurizer)
        self.assertListEqual(classifier.classifier.coef_.tolist(), coeffs)
        self.assertListEqual(classifier.classifier.intercept_.tolist(),
                             intercept)
        self.assertDictEqual(classifier.config.to_dict(), config)

    def test_should_get_intent_after_deserialization(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a cup of tea
- i want two cups of tea please
- can you prepare one cup of tea ?

---
type: intent
name: MakeCoffee
utterances:
- make me a cup of coffee please
- brew two cups of coffee
- can you prepare one cup of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        classifier = LogRegIntentClassifier().fit(dataset)
        classifier.persist(self.tmp_file_path)

        # When
        builtin_entity_parser = BuiltinEntityParser.build(language="en")
        custom_entity_parser = CustomEntityParser.build(
            dataset, CustomEntityParserUsage.WITHOUT_STEMS)
        loaded_classifier = LogRegIntentClassifier.from_path(
            self.tmp_file_path,
            builtin_entity_parser=builtin_entity_parser,
            custom_entity_parser=custom_entity_parser)
        result = loaded_classifier.get_intent("Make me two cups of tea")

        # Then
        expected_intent = "MakeTea"
        self.assertEqual(expected_intent, result[RES_INTENT_NAME])

    def test_should_be_serializable_into_bytearray(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a cup of tea
- i want two cups of tea please
- can you prepare one cup of tea ?

---
type: intent
name: MakeCoffee
utterances:
- make me a cup of coffee please
- brew two cups of coffee
- can you prepare one cup of coffee""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        intent_classifier = LogRegIntentClassifier().fit(dataset)

        # When
        intent_classifier_bytes = intent_classifier.to_byte_array()
        custom_entity_parser = CustomEntityParser.build(
            dataset, CustomEntityParserUsage.WITHOUT_STEMS)
        builtin_entity_parser = BuiltinEntityParser.build(language="en")
        loaded_classifier = LogRegIntentClassifier.from_byte_array(
            intent_classifier_bytes,
            builtin_entity_parser=builtin_entity_parser,
            custom_entity_parser=custom_entity_parser)
        result = loaded_classifier.get_intent("make me two cups of tea")

        # Then
        expected_intent = "MakeTea"
        self.assertEqual(expected_intent, result[RES_INTENT_NAME])

    @patch("snips_nlu.intent_classifier.log_reg_classifier"
           ".build_training_data")
    def test_empty_vocabulary_should_fit_and_return_none_intent(
            self, mocked_build_training):
        # Given
        language = LANGUAGE_EN
        dataset = {
            "entities": {
                "dummy_entity_1": {
                    "automatically_extensible": True,
                    "use_synonyms": False,
                    "data": [
                        {
                            "value": "...",
                            "synonyms": [],
                        }
                    ],
                    "matching_strictness": 1.0
                }
            },
            "intents": {
                "dummy_intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "...",
                                    "slot_name": "dummy_slot_name",
                                    "entity": "dummy_entity_1"
                                }
                            ]
                        }
                    ]
                }
            },
            "language": language
        }

        text = " "
        noise_size = 6
        utterances = [text] + [text] * noise_size
        utterances = [text_to_utterance(t) for t in utterances]
        labels = [0] + [1] * noise_size
        intent_list = ["dummy_intent_1", None]
        mocked_build_training.return_value = utterances, labels, intent_list

        # When / Then
        intent_classifier = LogRegIntentClassifier().fit(dataset)
        intent = intent_classifier.get_intent("no intent there")
        self.assertEqual(None, intent)

    def test_log_activation_weights(self):
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: intent1
utterances:
  - foo bar

---
type: intent
name: intent2
utterances:
  - lorem ipsum""")
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        intent_classifier = LogRegIntentClassifier().fit(dataset)

        text = "yo"
        utterances = [text_to_utterance(text)]
        x = intent_classifier.featurizer.transform(utterances)[0]

        # When
        log = intent_classifier.log_activation_weights(text, x, top_n=42)

        # Then
        self.assertIsInstance(log, str)
        self.assertIn("Top 42", log)
