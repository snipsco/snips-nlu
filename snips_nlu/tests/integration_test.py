# coding=utf-8
from __future__ import unicode_literals, print_function

import unittest

from future.utils import iteritems
from snips_nlu_metrics import (compute_cross_val_metrics,
                               compute_cross_val_nlu_metrics)
from snips_nlu_rust.nlu_engine import NLUEngine as InferenceEngine

from snips_nlu.constants import LANGUAGE_EN
from snips_nlu.nlu_engine.nlu_engine import SnipsNLUEngine as TrainingEngine
from snips_nlu.tests.utils import PERFORMANCE_DATASET_PATH, SnipsTest
from snips_nlu.tokenization import tokenize_light

INTENT_CLASSIFICATION_THRESHOLD = 0.8
SLOT_FILLING_THRESHOLD = 0.6

SKIPPED_DATE_PREFIXES = {"at", "in", "for", "on"}


class IntegrationTestSnipsNLUEngine(SnipsTest):
    def test_pure_python_engine_performance(self):
        # Given
        dataset_path = PERFORMANCE_DATASET_PATH

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

    def test_python_rust_engine_performance(self):
        # Given
        dataset_path = PERFORMANCE_DATASET_PATH

        # When
        results = compute_cross_val_nlu_metrics(
            dataset_path,
            training_engine_class=TrainingEngine,
            inference_engine_class=InferenceEngine,
            nb_folds=5,
            train_size_ratio=1.0,
            slot_matching_lambda=None,
            progression_handler=None)

        # Then
        self.check_metrics(results)

    def check_metrics(self, results):
        for intent_name, intent_metrics in iteritems(results["metrics"]):
            if intent_name is None or intent_name == "null":
                continue
            classification_precision = intent_metrics["intent"]["precision"]
            classification_recall = intent_metrics["intent"]["recall"]
            self.assertGreaterEqual(
                classification_precision, INTENT_CLASSIFICATION_THRESHOLD,
                "Intent classification precision is too low (%.3f) for intent "
                "'%s'" % (classification_precision, intent_name))
            self.assertGreaterEqual(
                classification_recall, INTENT_CLASSIFICATION_THRESHOLD,
                "Intent classification recall is too low (%.3f) for intent "
                "'%s'" % (classification_recall, intent_name))
            for slot_name, slot_metrics in iteritems(intent_metrics["slots"]):
                precision = slot_metrics["precision"]
                recall = slot_metrics["recall"]
                self.assertGreaterEqual(
                    precision, SLOT_FILLING_THRESHOLD,
                    "Slot precision is too low (%.3f) for slot '%s' of intent "
                    "'%s'" % (precision, slot_name, intent_name))
                self.assertGreaterEqual(
                    recall, SLOT_FILLING_THRESHOLD,
                    "Slot recall is too low (%.3f) for slot '%s' of intent "
                    "'%s'" % (recall, slot_name, intent_name))


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
