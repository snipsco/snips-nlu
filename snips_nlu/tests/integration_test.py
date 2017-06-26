# coding=utf-8
from __future__ import unicode_literals

import io
import json
import os
import unittest
from copy import deepcopy

from snips_nlu.constants import TEXT, DATA, PARSED_INTENT, INTENT_NAME, \
    SLOT_NAME, MATCH_RANGE, PARSED_SLOTS, INTENTS, UTTERANCES, CUSTOM_ENGINE, \
    ENGINE_TYPE
from snips_nlu.languages import Language
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.tests.utils import TEST_PATH


class IntegrationTestSnipsNLUEngine(unittest.TestCase):
    def test_engine_performance(self):
        # Given
        dataset_path = os.path.join(TEST_PATH, "resources",
                                    "performance_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        k_fold_batches = create_k_fold_batches(dataset, k=3)
        parsing_results = []
        for (train_dataset, test_utterances) in k_fold_batches:
            engine = SnipsNLUEngine(Language.EN).fit(train_dataset)
            for intent_name, utterance in test_utterances:
                text = "".join([chunk[TEXT] for chunk in utterance[DATA]])
                result = engine.parse(text)
                if result[PARSED_INTENT] is None:
                    parsing_results.append(0.0)
                else:
                    if result[PARSED_INTENT][INTENT_NAME] != intent_name:
                        parsing_results.append(0.0)
                    else:
                        parsing_result = 1.0
                        slot_chunks = [chunk for chunk in utterance[DATA]
                                       if SLOT_NAME in chunk]
                        for chunk in slot_chunks:
                            chunk_range = [chunk[MATCH_RANGE]["start"],
                                           chunk[MATCH_RANGE]["end"]]
                            no_matching_slot = all(s[SLOT_NAME] != chunk[
                                SLOT_NAME] or s[MATCH_RANGE] != chunk_range for
                                                   s in result[PARSED_SLOTS])
                            if no_matching_slot:
                                parsing_result = 0.0
                                break
                        parsing_results.append(parsing_result)
        accuracy = sum(parsing_results) / len(parsing_results)
        self.assertGreaterEqual(accuracy, 0.35)


def create_k_fold_batches(dataset, k):
    utterances = [
        (intent_name, utterance, i)
        for intent_name, intent_data in dataset[INTENTS].iteritems()
        for i, utterance in enumerate(intent_data[UTTERANCES])
    ]
    utterances = sorted(utterances, key=lambda u: u[2])
    utterances = [(intent_name, utterance) for (intent_name, utterance, _) in
                  utterances]
    nb_utterances = len(utterances)
    k_fold_batches = []
    batch_size = nb_utterances / k
    for batch_index in xrange(k):
        test_start = batch_index * batch_size
        test_end = (batch_index + 1) * batch_size
        train_utterances = utterances[0:test_start] + utterances[test_end:]
        test_utterances = utterances[test_start: test_end]
        train_dataset = deepcopy(dataset)
        train_dataset[INTENTS] = dict()
        for intent_name, utterance in train_utterances:
            if intent_name not in train_dataset[INTENTS]:
                train_dataset[INTENTS][intent_name] = {
                    ENGINE_TYPE: CUSTOM_ENGINE,
                    UTTERANCES: []
                }
            train_dataset[INTENTS][intent_name][UTTERANCES].append(
                deepcopy(utterance))
        k_fold_batches.append((train_dataset, test_utterances))
    return k_fold_batches
