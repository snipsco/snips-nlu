# coding=utf-8
from __future__ import print_function, unicode_literals

import json
from builtins import range, str

from future.utils import iteritems
from snips_nlu_metrics import compute_cross_val_metrics

from snips_nlu.constants import LANGUAGE_EN
from snips_nlu.nlu_engine.nlu_engine import (
    SnipsNLUEngine, SnipsNLUEngine as TrainingEngine)
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.tests.utils import PERFORMANCE_DATASET_PATH, SnipsTest

INTENT_CLASSIFICATION_THRESHOLD = 0.95
SLOT_FILLING_THRESHOLD = 0.85

SKIPPED_DATE_PREFIXES = {"at", "in", "for", "on"}


class IntegrationTestSnipsNLUEngine(SnipsTest):
    def test_pure_python_engine_performance(self):
        # Given
        dataset_path = str(PERFORMANCE_DATASET_PATH)

        # When
        results = compute_cross_val_metrics(
            dataset_path,
            engine_class=TrainingEngine,
            nb_folds=5,
            train_size_ratio=1.0,
            slot_matching_lambda=_slot_matching_lambda,
            progression_handler=None)

        # Then
        self.check_metrics(results)

    def check_metrics(self, results):
        for intent_name, intent_metrics in iteritems(results["metrics"]):
            if intent_name is None or intent_name == "null":
                continue
            classification_f1 = intent_metrics["intent"]["f1"]
            self.assertGreaterEqual(
                classification_f1, INTENT_CLASSIFICATION_THRESHOLD,
                "Intent classification f1 score is too low (%.3f) for intent "
                "'%s'" % (classification_f1, intent_name))
            for slot_name, slot_metrics in iteritems(intent_metrics["slots"]):
                slot_f1 = slot_metrics["f1"]
                self.assertGreaterEqual(
                    slot_f1, SLOT_FILLING_THRESHOLD,
                    "Slot f1 score is too low (%.3f) for slot '%s' of intent "
                    "'%s'" % (slot_f1, slot_name, intent_name))

    def test_nlu_engine_training_is_deterministic(self):
        # We can't write a test to ensure the NLU training is always the same
        # instead we train the NLU 10 times and check the learnt parameters.
        # It does not bring any guarantee but it might alert us once in a while

        # Given
        num_runs = 10
        random_state = 42

        with PERFORMANCE_DATASET_PATH.open("r") as f:
            dataset = json.load(f)

        ref_log_reg, ref_crfs = None, None
        for _ in range(num_runs):
            engine = SnipsNLUEngine(random_state=random_state).fit(dataset)
            log_reg = _extract_log_reg(engine)
            crfs = _extract_crfs(engine)

            if ref_log_reg is None:
                ref_log_reg = log_reg
                ref_crfs = crfs
            else:
                self.assertDictEqual(ref_log_reg, log_reg)
                self.assertDictEqual(ref_crfs, crfs)


def _slot_matching_lambda(lhs_slot, rhs_slot):
    lhs_value = lhs_slot["text"]
    rhs_value = rhs_slot["rawValue"]
    if lhs_slot["entity"] != "snips/datetime":
        return lhs_value == rhs_value
    else:
        # Allow fuzzy matching when comparing datetimes
        lhs_tokens = tokenize_light(lhs_value, LANGUAGE_EN)
        rhs_tokens = tokenize_light(rhs_value, LANGUAGE_EN)
        if lhs_tokens and lhs_tokens[0].lower() in SKIPPED_DATE_PREFIXES:
            lhs_tokens = lhs_tokens[1:]
        if rhs_tokens and rhs_tokens[0].lower() in SKIPPED_DATE_PREFIXES:
            rhs_tokens = rhs_tokens[1:]
        return lhs_tokens == rhs_tokens


def _extract_log_reg(engine):
    log_reg = dict()
    intent_classifier = engine.intent_parsers[1].intent_classifier
    log_reg["intent_list"] = intent_classifier.intent_list
    log_reg["coef"] = intent_classifier.classifier.coef_.tolist()
    log_reg["intercept"] = intent_classifier.classifier.intercept_.tolist()
    return log_reg


def _extract_crfs(engine):
    crfs = dict()
    slot_fillers = engine.intent_parsers[1].slot_fillers
    for intent, slot_filler in iteritems(slot_fillers):
        crfs[intent] = {
            "state_features": slot_filler.crf_model.state_features_,
            "transition_features": slot_filler.crf_model.transition_features_
        }
    return crfs
