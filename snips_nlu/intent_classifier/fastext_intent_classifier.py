# coding=utf-8
from __future__ import division, unicode_literals

import json
import logging
from itertools import product
from operator import itemgetter

import numpy as np
from fastText import load_model, train_supervised
from future.builtins import str, zip
from future.utils import iteritems
from pathlib import Path

from snips_nlu.constants import (DATA, ENTITY,
                                 ENTITY_KIND, TEXT, ROOT_PATH)
from snips_nlu.dataset import get_text_from_chunks, validate_and_format_dataset
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.featurizer import (
    _builtin_entity_to_feature, _entity_name_to_feature,
    _normalize_stem)
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    build_training_data)
from snips_nlu.pipeline.configs.intent_classifier import (
    FastTextIntentClassifierConfig)
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.result import intent_classification_result
from snips_nlu.utils import (
    check_random_state, json_string, temp_dir)

logger = logging.getLogger(__name__)


class FastTextIntentClassifier(IntentClassifier):
    unit_name = "fast_text_intent_classifier"
    config_type = FastTextIntentClassifierConfig

    def __init__(self, config, **shared):
        super(FastTextIntentClassifier, self).__init__(config, **shared)
        self.language = None
        self.model = None

    def fit(self, dataset, force_retrain=True):
        dataset = validate_and_format_dataset(dataset)
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        self.language = dataset["language"]

        random_state = check_random_state(self.config.random_seed)
        data_augmentation_config = self.config.data_augmentation_config

        utterances, classes, intent_list = build_training_data(
            dataset, self.language, data_augmentation_config,
            random_state)

        fastext_dataset = self._dataset_to_fasttext_file(
            utterances, classes, intent_list, random_state)

        num_utterances = len(fastext_dataset)
        if num_utterances < 2:
            raise ValueError("Dataset should have at least 2 utterances")

        train_id = min(num_utterances - 1,
                       int(num_utterances * self.config.validation_ratio))
        val_dataset = fastext_dataset[:train_id]
        train_dataset = fastext_dataset[train_id:]

        with temp_dir() as tmp_dir:
            train_dataset_path = tmp_dir / "train_dataset.txt"
            val_dataset_path = tmp_dir / "val_dataset.txt"
            with train_dataset_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(train_dataset))
            with val_dataset_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(val_dataset))

            self.model, best_f1, best_params = grid_search(
                train_dataset_path, val_dataset_path,
                self.config.learning_rates, self.config.embedding_dims,
                self.config.epochs, self.config.ngram_lengths)
            print "Best model F1: %s" % best_f1
            print "Best model params: %s" % best_params
            logger.debug("Best model F1: %s", best_f1)
            logger.debug("Best model params: %s", best_params)
            # TODO: quantize the model
            # self.model.quantize()
        return self

    def get_intents_probs(self, text):
        features = self._featurize_utterance({"data": [{"text": text}]})
        text_input = " ".join(features)
        num_labels = len(self.model.get_labels())
        labels, probs = self.model.predict(text_input, k=num_labels)
        none_label = create_label(None)
        labels = [l.lstrip("__label__") if l != none_label else None
                  for l in labels]
        return sorted(zip(labels, probs), key=itemgetter(1), reverse=True)

    def get_intent(self, text, intents_filter=None):
        intents_probs = self.get_intents_probs(text)
        best_intent, best_prob = intents_probs[0]
        if intents_filter is not None and best_intent not in intents_filter:
            return None
        return intent_classification_result(best_intent, best_prob)

    @property
    def fitted(self):
        return self.model is not None

    def persist(self, path):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)

        classifier_name = "intent_classifier"
        model_path = path / classifier_name

        self.model.save_model(str(model_path))

        self_as_dict = {
            "config": self.config.to_dict(),
            "model": classifier_name,
            "language": self.language,
        }

        classifier_path = path / "intent_classifier.json"
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

        model = model.pop("model")
        if model is not None:
            model = load_model(str(path / model))
        config = model.pop("config")
        self = cls(config, **shared)
        self.model = model

        for attribute, value in iteritems(model):
            setattr(self, attribute, value)

        return self

    def _dataset_to_fasttext_file(self, utterances, classes,
                                  class_to_intent, random_state):
        queries = []
        for u, u_class in zip(utterances, classes):
            intent_id = class_to_intent[u_class]
            features = self._featurize_utterance(u)
            # features = sorted(features)  # For testing
            queries.append(" ".join(features) + " " + create_label(intent_id))
        return random_state.permutation(queries)

    def _featurize_utterance(self, utterance):
        features = []
        utterance_text = get_text_from_chunks(utterance[DATA])
        utterance_tokens = tokenize_light(utterance_text, self.language)
        if not utterance_tokens:
            return features

        # word_clusters_features = _get_word_cluster_features(
        #     utterance_tokens, self.config.word_clusters_name, self.language)

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
        # if word_clusters_features:
        #     features += word_clusters_features
        return features


def create_label(text):
    return "__label__%s" % text


def compute_f1(model, validation_examples, validation_labels):
    p, r = compute_precision_recall(
        model, validation_examples, validation_labels)
    f1 = 2 * (p * r) / (p + r)
    return np.nan_to_num(f1)


def compute_precision_recall(model, validation_examples, validation_labels):
    label_to_ix = {v: k for k, v in enumerate(set(validation_labels))}
    num_labels = len(label_to_ix)
    predicted_labels, _ = model.predict(validation_examples, k=1)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=np.int)
    for pred_label, val_label in zip(predicted_labels, validation_labels):
        val_ix = label_to_ix[val_label]
        pred_ix = label_to_ix.get(pred_label)
        if pred_ix is None:  # Predicted label is not in the val set
            zero_row = np.zeros((1, confusion_matrix.shape[1]), dtype=np.int)
            confusion_matrix = np.vstack((confusion_matrix, zero_row))

            zero_col = np.zeros((confusion_matrix.shape[0], 1), dtype=np.int)
            confusion_matrix = np.hstack((confusion_matrix, zero_col))
            label_to_ix[pred_label] = len(label_to_ix)
            pred_ix = label_to_ix[pred_label]
        confusion_matrix[val_ix, pred_ix] += 1.0

    tp = np.diag(confusion_matrix)
    precision_matrix = np.nan_to_num(tp / confusion_matrix.sum(axis=0))
    recall_matrix = np.nan_to_num(tp / confusion_matrix.sum(axis=1))
    return precision_matrix, recall_matrix


def grid_search(train_path, validation_path, learning_rates, embedding_dims,
                epochs, ngram_lengths):
    best_model = None
    best_f1 = None
    best_params = None

    validation_examples = None
    validation_labels = None


    pretrain_vector_path = str(
        ROOT_PATH / "fastext_snips_dim_10_lr_0.1_epoch_1000_min_3_max_6_loss_0.22.vec")

    for lr, dim, epoch, n in product(learning_rates, embedding_dims, epochs,
                                     ngram_lengths):
        model = train_supervised(
            str(train_path), lr=lr,
            dim=dim,
            epoch=epoch, minCount=1,
            wordNgrams=n, minCountLabel=0, minn=0, maxn=0, neg=5,
            loss="softmax", bucket=2000000, thread=4, lrUpdateRate=100,
            t=1e-4, label="__label__", verbose=2,
            pretrainedVectors=pretrain_vector_path,
        )

        if validation_examples is None:
            with validation_path.open("r", encoding="utf-8") as f:
                validation_examples, validation_labels = model.get_line(
                    [l.strip() for l in f.readlines()])
                validation_examples = [
                    " ".join(e) for e in validation_examples]
                validation_labels = [l[0] for l in validation_labels]

        f1 = compute_f1(model, validation_examples, validation_labels).mean()

        if f1 > best_f1:
            best_model = model
            best_params = {
                "lr": lr,
                "dim": dim,
                "epoch": epoch,
                "wordNgrams": n
            }
            best_f1 = f1
    return best_model, best_f1, best_params
