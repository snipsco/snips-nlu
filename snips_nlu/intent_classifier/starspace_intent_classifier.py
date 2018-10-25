# coding=utf-8
from __future__ import unicode_literals

import json
from copy import deepcopy
from itertools import combinations
from operator import itemgetter
from subprocess import check_call

import numpy as np
from future.builtins import range, str, zip
from future.utils import iteritems, viewvalues
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from snips_nlu.constants import (DATA, ENTITY,
                                 ENTITY_KIND, INTENTS, ROOT_PATH, TEXT,
                                 UTTERANCES)
from snips_nlu.dataset import get_text_from_chunks, validate_and_format_dataset
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.featurizer import (
    _builtin_entity_to_feature, _entity_name_to_feature,
    _get_word_cluster_features, _normalize_stem)
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    build_training_data)
from snips_nlu.pipeline.configs import (
    StarSpaceIntentClassifierConfig)
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.result import intent_classification_result
from snips_nlu.utils import (
    check_random_state, json_string, temp_dir)

"""
TODO: just save the labels embedding and the model at inference use the label
 embedding rather than test script

TODO: add a None intent
"""


class StarSpaceIntentClassifier(IntentClassifier):
    unit_name = "starspace_intent_classifier"
    config_type = StarSpaceIntentClassifierConfig

    def __init__(self, config, **shared):
        super(StarSpaceIntentClassifier, self).__init__(config, **shared)
        self.starspace_binary_path = str(ROOT_PATH / "Starspace" / 'starspace')

        self.labels_embeddings = None
        self.labels = None
        self.language = None
        self.embedding = None

    def fit(self, dataset):
        dataset = validate_and_format_dataset(dataset)
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        self.language = dataset["language"]

        # TODO: data augmentation
        random_state = check_random_state(self.config.random_seed)
        data_augmentation_config = self.config.data_augmentation_config

        # Cross intents
        max_intents = min(len(dataset["intents"]), self.config.max_intents)
        multi_intent_dataset = create_multi_intent_dataset(
            dataset, max_intents, random_state)

        utterances, classes, intent_list = build_training_data(
            multi_intent_dataset, self.language, data_augmentation_config,
            random_state)

        intent_list[-1] = (intent_list[-1],)  # Trick so that None is also a
        # tuple

        starspace_dataset = self._dataset_to_starspace_file(
            utterances, classes, intent_list, random_state)

        num_utterances = len(starspace_dataset)
        if num_utterances < 2:
            raise ValueError("Dataset should have at least 2 utterances")

        train_id = min(num_utterances - 1,
                       int(num_utterances * self.config.validation_ratio))
        val_dataset = starspace_dataset[:train_id]
        train_dataset = starspace_dataset[train_id:]
        self.base_labels = list(dataset[INTENTS])

        max_neg_samples = 2 * len(dataset[INTENTS])
        with temp_dir() as tmp_dir:
            train_dataset_path = tmp_dir / "train_dataset.txt"
            val_dataset_path = tmp_dir / "val_dataset.txt"
            with train_dataset_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(train_dataset))
            with val_dataset_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(val_dataset))

            model_path = tmp_dir / "model"
            cmd = [
                "%s" % self.starspace_binary_path, "train",
                "-trainFile", str(train_dataset_path),
                "-validationFile", str(val_dataset_path),
                "-validationPatience", str(self.config.validation_patience),
                "-model", str(model_path),
                "-dim", str(self.config.embedding_dim),
                "-epoch", str(100),
                "-thread", str(self.config.n_threads),
                "-negSearchLimit", str(self.config.neg_search_limit),
                "-maxNegSamples", str(max_neg_samples)
            ]
            check_call(cmd)
            embedding_path = tmp_dir / "model.tsv"
            self.embedding = load_embedding(embedding_path)

        all_labels = list(dataset["intents"])
        self.labels = list()
        self.labels_embeddings = []
        for combination_size in range(1, self.config.max_intents + 1):
            for labels in combinations(all_labels, combination_size):
                labels_embedding = [
                    self.embedding[create_label(l)] for l in labels]
                self.labels_embeddings.append(
                    np.array(labels_embedding).sum(axis=0))
                self.labels.append(labels)

        # Add the embedding for the None intent
        self.labels.append((None,))
        self.labels_embeddings.append(self.embedding[create_label(None)])

        self.labels_embeddings = np.array(self.labels_embeddings)
        return self

    def get_intents_probs(self, text):
        features = self._featurize_utterance({"data": [{"text": text}]})
        features_embeddings = [self.embedding[f] for f in features
                               if f in self.embedding]
        text_embeddings = np.array(features_embeddings).sum(axis=0)

        similarities = cosine_similarity(
            self.labels_embeddings, text_embeddings.reshape(1, -1)).flatten()
        probs = (similarities + 1.0) / 2.0
        labels = ["+".join(label) if label != (None,) else None
                  for label in self.labels]
        intent_ix_and_probs = zip(labels, probs)
        return sorted(intent_ix_and_probs, key=itemgetter(1), reverse=True)

    def get_intent(self, text, intents_filter=None):
        intents_probs = self.get_intents_probs(text)
        best_intent, best_prob = intents_probs[0]
        if intents_filter is not None and best_intent not in intents_filter:
            return None
        return intent_classification_result(best_intent, best_prob)

    @property
    def fitted(self):
        return self.embedding is not None

    def persist(self, path):
        path = Path(path)
        self_as_dict = {
            "config": self.config.to_dict(),
            "labels_embeddings": self.labels_embeddings.tolist(),
            "labels": self.labels,
            "language": self.language,
            "embedding": self.embedding,
        }

        classifier_path = path / "intent_classifier.json"

        path.mkdir(parents=True)
        with classifier_path.open("w", encoding="utf-8") as f:
            f.write(json_string(self_as_dict))

        # Metadata
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)

        model_path = path / "intent_classifier.json"
        if not model_path.exists():
            raise OSError("Missing intent classifier model file: %s"
                          % model_path.name)

        with model_path.open("r", encoding="utf-8") as f:
            model = json.load(f)

        config = model.pop("config")
        self = cls(config, **shared)

        label_embeddings = model.pop("labels_embeddings")
        if label_embeddings is not None:
            label_embeddings = np.array(label_embeddings)
            self.labels_embeddings = label_embeddings

        for attribute, value in iteritems(model):
            setattr(self, attribute, value)

        return self

    def _dataset_to_starspace_file(self, utterances, classes,
                                   class_to_intents, random_state):
        queries = []
        for u, u_class in zip(utterances, classes):
            intent_ids = class_to_intents[u_class]
            joined_labels = " ".join(
                create_label(i) for i in sorted(intent_ids))
            features = self._featurize_utterance(u)
            # features = sorted(features)  # For testing
            queries.append(" ".join(features) + " " + joined_labels)
        return random_state.permutation(queries)

    def _featurize_utterance(self, utterance):
        features = []
        utterance_text = get_text_from_chunks(utterance[DATA])
        utterance_tokens = tokenize_light(utterance_text, self.language)
        if not utterance_tokens:
            return features

        word_clusters_features = _get_word_cluster_features(
            utterance_tokens, self.config.word_clusters_name, self.language)
        normalized_stemmed_tokens = [
            _normalize_stem(t, self.language, self.config.use_stemming)
            for t in utterance_tokens
        ]

        custom_entities = self.custom_entity_parser.parse(
            " ".join(normalized_stemmed_tokens))
        if self.config.unknown_words_replacement_string:
            custom_entities = [
                e for e in custom_entities
                if e["value"] != self.config.unknown_words_replacement_string
            ]
        custom_entities_features = [
            _entity_name_to_feature(e[ENTITY_KIND], self.language)
            for e in custom_entities]

        builtin_entities = self.builtin_entity_parser.parse(
            utterance_text, use_cache=True)
        builtin_entities_features = [
            _builtin_entity_to_feature(ent[ENTITY_KIND], self.language)
            for ent in builtin_entities
        ]

        # We remove values of builtin slots from the utterance to avoid
        # learning specific samples such as '42' or 'tomorrow'
        filtered_normalized_stemmed_tokens = []
        for chunk in utterance[DATA]:
            if ENTITY in chunk and is_builtin_entity(chunk[ENTITY]):
                continue
            filtered_tokens = tokenize_light(chunk[TEXT], self.language)
            filtered_normalized_stemmed_tokens += [
                _normalize_stem(t, self.language, True)
                for t in filtered_tokens]

        features = filtered_normalized_stemmed_tokens
        if builtin_entities_features:
            features += builtin_entities_features
        if custom_entities_features:
            features += custom_entities_features
        if word_clusters_features:
            features += word_clusters_features
        return features


def load_embedding(embedding_path):
    embedding_path = Path(embedding_path)
    embedding = dict()
    with embedding_path.open("r", encoding="utf-8") as f:
        for l in f:
            split = l.strip().split("\t")
            embedding[split[0]] = [float(n) for n in split[1:]]
    return embedding


def create_label(text):
    return "__label__%s" % text


def create_multi_intent_dataset(dataset, max_intents, random_state,
                                junction_formulas=["and"]):
    # TODO: junction_formulas should not be optional but set per language
    multi_intent_dataset = deepcopy(dataset)
    intents = {
        (intent_name,): intent
        for intent_name, intent in iteritems(multi_intent_dataset[INTENTS])
    }

    if max_intents < 1:
        raise ValueError("max_intents must be >= 1")
    if max_intents == 1:
        multi_intent_dataset[INTENTS] = intents
        return multi_intent_dataset

    num_queries = {
        intent_id: len(intent[UTTERANCES])
        for intent_id, intent in iteritems(multi_intent_dataset[INTENTS])
    }
    total_queries = sum(viewvalues(num_queries))

    for combination_size in range(2, max_intents + 1):
        all_intents_combinations = list(
            combinations(sorted(dataset["intents"]), combination_size))
        all_intents_combinations_ix = [
            i for i in range(len(all_intents_combinations))]

        for _ in range(total_queries):
            combination_ix = random_state.choice(all_intents_combinations_ix)
            combination = all_intents_combinations[combination_ix]
            selected_utterances_ix = [
                (i, random_state.choice(num_queries[i])) for i in combination]
            selected_utterances = [
                multi_intent_dataset[INTENTS][i][UTTERANCES][ix]
                for i, ix in selected_utterances_ix
            ]
            permuted_utterances = random_state.permutation(selected_utterances)
            mixed_utterance = deepcopy(permuted_utterances[0])
            for u in permuted_utterances[1:]:
                junction_utterances = [
                    {"text": " %s " % random_state.choice(junction_formulas)}
                ]
                mixed_utterance["data"] += junction_utterances
                mixed_utterance["data"] += u["data"]
            if combination not in intents:
                intents[combination] = {UTTERANCES: []}
            intents[combination][UTTERANCES].append(mixed_utterance)

    multi_intent_dataset[INTENTS] = intents
    return multi_intent_dataset
