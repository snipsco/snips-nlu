import io
import json
import logging
import unittest
from datetime import datetime

from snips_nlu_metrics import Engine, compute_cross_val_metrics

from snips_nlu.cli.metrics import _match_trimmed_values
from snips_nlu.cli.utils import set_nlu_logger
from snips_nlu.common.io_utils import temp_dir
from snips_nlu.constants import ROOT_PATH
from snips_nlu.dataset import Dataset
from snips_nlu.intent_classifier.paraphrase_classifier import (
    LogRegIntentClassifierWithParaphrase, remove_slots)
from snips_nlu.pipeline.configs.intent_classifier import (
    LogRegIntentClassifierWithParaphraseConfig, ParaphraseClassifierConfig)
from snips_nlu.result import parsing_result

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
remove_slots(ELECTROLUX_DATASET)


def make_engine_cls(intent_classifier_cls, config, shared):

    class IntentClassifierEngine(Engine):
        def __init__(self):
            self.cls = intent_classifier_cls
            self.config = config
            self.shared = shared
            self._clf = None

        def fit(self, dataset):
            self._clf = self.cls(self.config, **self.shared)
            self._clf.fit(dataset)
            return self

        def parse(self, text, intents_filter=None):
            return parsing_result(
                text,
                self._clf.get_intent(text, intents_filter=intents_filter),
                [],
            )

    return IntentClassifierEngine


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
            "lr": 5e-3,
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

        random_state = 220
        clf = LogRegIntentClassifierWithParaphrase(
            config=config, random_state=random_state, **shared)

        clf.fit(dataset)
        with temp_dir() as tmp:
            clf_path = tmp / "paraphrase_classifier"
            clf.persist(clf_path)
            new_clf = LogRegIntentClassifierWithParaphrase.from_path(
                clf_path, **shared)
        new_clf.get_intents("i'm here")

    def test_metrics(self):
        set_nlu_logger(logging.DEBUG)
        # Given
        dataset = ELECTROLUX_DATASET
        stamp = datetime.now()
        log_dir = ROOT_PATH / ".log" / str(stamp).replace(":", "_")
        output_dir = log_dir / "intent_classifier"
        random_state = 220

        shared = {
            "log_dir": log_dir,
            "output_dir": output_dir,
            "random_state": random_state,
        }
        validation_ratio = .1
        sentence_classifier_config = {
            "name": "mlp_intent_classifier",
            "hidden_sizes": [],
            "activation": "SELU",
            "dropout": .5,
        }
        optimizer_config = {
            "lr": 5e-3,
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


        engine_cls = make_engine_cls(
            LogRegIntentClassifierWithParaphrase, config, shared)
        metrics_args = dict(
            dataset=dataset,
            engine_class=engine_cls,
            slot_matching_lambda=_match_trimmed_values,
            seed=random_state,
        )
        metrics = compute_cross_val_metrics(**metrics_args)
        metrics_path = log_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f)
