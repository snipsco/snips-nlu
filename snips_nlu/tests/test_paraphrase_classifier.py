import io
import json
import logging
import unittest
from datetime import datetime

from snips_nlu.cli.utils import set_nlu_logger
from snips_nlu.common.io_utils import temp_dir
from snips_nlu.constants import ROOT_PATH
from snips_nlu.dataset import Dataset
from snips_nlu.intent_classifier.paraphrase_classifier import (
    LogRegIntentClassifierWithParaphrase, ParaphraseClassifier)
from snips_nlu.pipeline.configs.intent_classifier import (
    LogRegIntentClassifierWithParaphraseConfig, ParaphraseClassifierConfig)

TOY_DATASET = """
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
"""
TOY_DATASET = Dataset.from_yaml_files("en", [io.StringIO(TOY_DATASET)]).json

ELECTROLUX_PATH = ROOT_PATH / "dataset_electrolux.json"

with ELECTROLUX_PATH.open() as f:
    ELECTROLUX_DATASET = json.load(f)


class TestParaphraseClassifier(unittest.TestCase):
    def test_train(self):
        set_nlu_logger(logging.DEBUG)
        # Given
        dataset = ELECTROLUX_DATASET
        stamp = datetime.now()
        log_dir = ROOT_PATH / ".log" / str(stamp).replace(":", "_")
        output_dir = log_dir / "intent_classifier"

        shared = {
            "log_dir": log_dir,
            "output_dir": output_dir
        }
        validation_ratio = .1
        sentence_classifier_config = {
            "name": "mlp_intent_classifier",
            "hidden_sizes": [],
            "activation": "SELU",
            "dropout": .5,
        }
        optimizer_config = {
            "lr": 1e-3,
        }
        paraphrase_clf_config = ParaphraseClassifierConfig(
            sentence_classifier_config, )
        config = LogRegIntentClassifierWithParaphraseConfig(
            n_epochs=int(1e5),
            num_paraphrases=1,
            validation_ratio=validation_ratio,
            batch_size=64,
            paraphrase_classifier_config=paraphrase_clf_config,
            optimizer_config=optimizer_config,
        )

        random_state = 110
        clf = LogRegIntentClassifierWithParaphrase(
            config=config, random_state=random_state, **shared)

        clf.fit(dataset)
        with temp_dir() as tmp:
            clf_path = tmp / "paraphrase_classifier"
            clf.persist(clf_path)
            LogRegIntentClassifierWithParaphrase.from_path(
                clf_path, **shared)
