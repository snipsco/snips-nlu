import io
import logging
import unittest
from datetime import datetime

from snips_nlu.cli.utils import set_nlu_logger
from snips_nlu.constants import ROOT_PATH
from snips_nlu.dataset import Dataset
from snips_nlu.intent_classifier.paraphrase_classifier import (
    LogRegIntentClassifierWithParaphrase)
from snips_nlu.pipeline.configs.intent_classifier import (
    LogRegIntentClassifierWithParaphraseConfig, ParaphraseClassifierConfig)


class TestParaphraseClassifier(unittest.TestCase):
    def test_train(self):
        set_nlu_logger(logging.DEBUG)
        # Given
        dataset_stream = io.StringIO("""
---
type: intent
name: MakeTea
utterances:
- make me a [beverage_temperature:Temperature](hot) cup of tea
- make me [number_of_cups:snips/number](five) tea cups
- would you make [number_of_cups:snips/number](five) of [beverage_temperature:Temperature](very hot) tea please
- i need a cup of tea

---
type: intent
name: MakeCoffee
utterances:
- make me [number_of_cups:snips/number](one) cup of coffee please
- brew [number_of_cups:snips/number](two) cups of coffee
- can you make me some [beverage_temperature:Temperature](very hot) coffee please
- i think i'd like to get a coffee
""")
        stamp = datetime.now()
        log_dir = ROOT_PATH / ".log" / str(stamp).replace(":", "_")
        output_dir = log_dir / "intent_classifier"
        dataset = Dataset.from_yaml_files("en", [dataset_stream]).json
        shared = {
            "log_dir": log_dir,
            "output_dir": output_dir
        }
        validation_ratio = .4
        sentence_classifier_config = {
            "name": "mlp_intent_classifier",
            "hidden_sizes": [],
            "activation": "SELU",
            "dropout": .0,
        }
        optimizer_config = {
            "lr": 1e-3,
        }
        paraphrase_clf_config = ParaphraseClassifierConfig(
            sentence_classifier_config, )
        config = LogRegIntentClassifierWithParaphraseConfig(
            n_epochs=1000,
            num_paraphrases=3,
            validation_ratio=validation_ratio,
            batch_size=8,
            paraphrase_classifier_config=paraphrase_clf_config,
            optimizer_config=optimizer_config,
        )

        random_state = 110
        clf = LogRegIntentClassifierWithParaphrase(
            config=config, random_state=random_state, **shared)
        clf.fit(dataset)
