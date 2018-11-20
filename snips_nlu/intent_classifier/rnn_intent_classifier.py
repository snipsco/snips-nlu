# coding=utf-8
from __future__ import unicode_literals

import logging
from copy import deepcopy
from itertools import product

import numpy as np
from builtins import range, str
from fastText.FastText import _FastText
from future.utils import iteritems
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from tensorflow.python import Constant
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint)
from tensorflow.python.keras.layers import (Concatenate, Dense, Dropout,
                                            Embedding, LSTM, Masking)
from tensorflow.python.keras.utils import to_categorical

from snips_nlu.constants import DATA, ENTITY_KIND, INTENTS, LANGUAGE, \
    ROOT_PATH, UTTERANCES
from snips_nlu.dataset import get_text_from_chunks, validate_and_format_dataset
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    build_training_data)
from snips_nlu.pipeline.configs.intent_classifier import (
    RNNIntentClassifierConfig)
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.result import intent_classification_result
from snips_nlu.utils import check_random_state, temp_dir

logger = logging.getLogger(__name__)

EOS = "eeeoossss"
BOS = "bbbooosss"


def rnn_model(embedding, num_customs, num_builtins, num_classes, rnn_dim,
              num_layers, max_sequence_length, dropout=0.0,
              train_embedding=False):
    sequence_input = Input((max_sequence_length,))
    masked_input = Masking(input_shape=(max_sequence_length,))(sequence_input)
    embedded_input = Embedding(
        embedding.shape[0],
        embedding.shape[1],
        input_length=max_sequence_length,
        embeddings_initializer=Constant(embedding),
        trainable=train_embedding
    )(masked_input)

    rnn_output = None
    for i in range(1, num_layers + 1):
        return_sequences = True
        if rnn_output is None:
            rnn_output = embedded_input
        if i == num_layers:
            return_sequences = False

        # rnn = GRU(rnn_dim,
        #           activation='tanh',
        #           recurrent_activation='hard_sigmoid',
        #           use_bias=True,
        #           kernel_initializer='glorot_uniform',
        #           recurrent_initializer='orthogonal',
        #           bias_initializer='zeros',
        #           dropout=dropout,
        #           recurrent_dropout=0.,
        #           return_sequences=return_sequences)
        rnn_output = LSTM(rnn_dim, unit_forget_bias=True, dropout=dropout,
                          return_sequences=return_sequences)(rnn_output)

    dropped_rnn_output = Dropout(dropout)(rnn_output)
    custom_counts = Input((num_customs,))
    builtin_counts = Input((num_builtins,))
    dense_input = Concatenate()(
        [dropped_rnn_output, custom_counts, builtin_counts])
    outputs = Dense(num_classes, activation="softmax")(dense_input)
    model = Model(inputs=[sequence_input, custom_counts, builtin_counts],
                  outputs=outputs)
    return model


class RNNIntentClassifier(IntentClassifier):
    unit_name = "rnn_intent_classifier"
    config_type = RNNIntentClassifierConfig

    def __init__(self, config, **shared):
        super(RNNIntentClassifier, self).__init__(config, **shared)
        self.model = None
        self.intent_list = None

    @property
    def fitted(self):
        return self.model is not None

    def fit(self, dataset, force_retrain=True):
        # assert False  # Set a large number in the data augmentation
        logger.debug("Fitting LogRegIntentClassifier...")
        dataset = validate_and_format_dataset(dataset)
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        self.language = dataset[LANGUAGE]

        random_state = check_random_state(self.config.random_seed)

        train_dataset, val_dataset = _train_test_split(
            dataset, self.config.validation_ratio, random_state)

        data_augmentation_config = self.config.data_augmentation_config

        train_utterances, train_classes, self.intent_list = \
            build_training_data(train_dataset, self.language,
                                data_augmentation_config,
                                random_state)
        train_utterances = [
            get_text_from_chunks(u[DATA]) for u in train_utterances]
        train_labels = to_categorical(np.asarray(train_classes))

        val_utterances, val_classes, val_class_to_intent = build_training_data(
            val_dataset, self.language, data_augmentation_config, random_state)
        val_utterances = [
            get_text_from_chunks(u[DATA]) for u in val_utterances]
        val_labels = to_categorical(np.asarray(val_classes))

        if self.intent_list != val_class_to_intent:
            raise ValueError

        self._fit_tokenizer(train_utterances + val_utterances)
        self._fit_entity_counter(train_utterances)

        x_train = self._transform_utterances(train_utterances)
        train_shuffled_index = random_state.permutation(
            range(x_train[0].shape[0]))
        x_train = (
            x_train[0][train_shuffled_index],
            x_train[1][train_shuffled_index],
            x_train[2][train_shuffled_index]
        )
        y_train = train_labels[train_shuffled_index]

        x_val = self._transform_utterances(val_utterances)
        val_shuffled_index = random_state.permutation(
            range(x_val[0].shape[0]))
        x_val = (
            x_val[0][val_shuffled_index],
            x_val[1][val_shuffled_index],
            x_val[2][val_shuffled_index]
        )
        y_val = val_labels[val_shuffled_index]

        data = x_train, y_train, x_val, y_val

        self.model, best_1, best_config = grid_search(
            data, len(self.custom_mapping), len(self.builtin_mapping),
            self.tokenizer, self.config)
        print("Best params: %s - (%s F1)" % (best_config, best_1))

        return self

    def get_intent(self, text, intents_filter=None):
        features = self._transform_utterances([text])
        pred = self.model.predict(features)[0]
        intent_ix = np.argmax(pred)

        prob = float(pred[intent_ix])
        intent = self.intent_list[intent_ix]
        if intent is None:
            return None
        return intent_classification_result(intent, prob)

    def _fit_entity_counter(self, utterances):
        normalized_utterances = [
            " ".join(tokenize_light(u, self.language)) for u in utterances]
        builtins = set(b[ENTITY_KIND] for u in normalized_utterances
                       for b in self.builtin_entity_parser.parse(u))
        self.builtin_mapping = {ent: i for i, ent in enumerate(builtins)}
        customs = set(c[ENTITY_KIND] for u in normalized_utterances
                      for c in self.custom_entity_parser.parse(u))
        self.custom_mapping = {ent: i for i, ent in enumerate(customs)}
        return self

    def _fit_tokenizer(self, utterances):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(utterances + [BOS])
        sequences = self.tokenizer.texts_to_sequences(utterances)
        self.max_length = max(len(s) for s in sequences) + 1  # For BOS
        return self

    def _transform_utterances(self, utterances):
        processed_utterances = ["%s %s" % (BOS, u) for u in utterances]
        sequences = self.tokenizer.texts_to_sequences(processed_utterances)
        sequences = pad_sequences(sequences, maxlen=self.max_length)

        builtin_features = np.zeros(
            (len(utterances), len(self.builtin_mapping)))
        custom_features = np.zeros(
            (len(utterances), len(self.custom_mapping)))
        for i, u in enumerate(utterances):
            normalized_u = " ".join(tokenize_light(u, self.language))
            builtin_entities = self.builtin_entity_parser.parse(normalized_u)
            custom_entities = self.custom_entity_parser.parse(normalized_u)
            for b in builtin_entities:
                builtin_features[
                    i, self.builtin_mapping[b[ENTITY_KIND]]] += 1.0
            for c in custom_entities:
                custom_features[i, self.custom_mapping[c[ENTITY_KIND]]] += 1.0

        return (sequences, custom_features, builtin_features)

    def persist(self):
        pass

    @classmethod
    def from_path(cls, path, **shared_resources):
        pass


def load_embedding(embedding_path, tokenizer):
    fasttext = _FastText(str(embedding_path))
    embedding_dim = fasttext.get_dimension()
    max_index = max(tokenizer.index_word)
    embedding = np.zeros((max_index + 1, embedding_dim))
    for i, w in iteritems(tokenizer.index_word):
        embedding[i - 1] = fasttext.get_word_vector(w)
    return embedding


class F1(Callback):
    def __init__(self, training_data):
        super(F1, self).__init__()
        self.training_data = training_data

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or dict()
        x_val = self.validation_data[:3]
        y_val = self.validation_data[3]

        probs = self.model.predict(x_val)
        y_pred = np.zeros_like(probs)
        pred_ix = np.argmax(probs, axis=1)
        y_pred[np.arange(y_pred.shape[0]), pred_ix] = 1.0
        f1 = f1_score(y_val[:-1], y_pred[:-1], average=None).mean()
        logs["val_f1"] = f1
        print('\nValidation F1: {}'.format(f1))

        x_train, y_val = self.training_data
        probs = self.model.predict(x_train)
        y_pred = np.zeros_like(probs)
        pred_ix = np.argmax(probs, axis=1)
        y_pred[np.arange(y_pred.shape[0]), pred_ix] = 1.0
        f1 = f1_score(y_val[:-1], y_pred[:-1], average=None).mean()
        logs["train_f1"] = f1
        print('Train F1: {}\n'.format(f1))


def grid_search(data, num_customs, num_builtins, tokenizer, config):
    x_train, y_train, x_val, y_val = data
    best_model, best_f1, best_config = None, None, None

    embeddings = config.embeddings
    rnn_dims = config.rnn_dims
    batch_sizes = config.batch_sizes
    num_layers = config.num_layers
    optimizers = config.optimizers
    dropouts = config.dropouts

    for (
            embedding, rnn_dim, batch_size, num_layer, optimizer,
            dropout) in product(
        embeddings, rnn_dims, batch_sizes, num_layers, optimizers,
        dropouts):
        config = {
            "embeddings": [embedding],
            "rnn_dims": [rnn_dim],
            "optimizers": [optimizer],
            "num_layers": [num_layer],
            "batch_sizes": [batch_size],
            "dropouts": [dropout]
        }
        print("Params: %s" % config)

        embedding_path = ROOT_PATH / embedding
        embedding = load_embedding(embedding_path, tokenizer)

        first_model = rnn_model(
            embedding,
            num_customs,
            num_builtins,
            y_train.shape[1],
            rnn_dim,
            num_layer,
            x_train[0].shape[1],
            dropout,
            train_embedding=False
        )

        second_model = rnn_model(
            embedding,
            num_customs,
            num_builtins,
            y_train.shape[1],
            rnn_dim,
            num_layer,
            x_train[0].shape[1],
            dropout,
            train_embedding=True
        )

        first_model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer
        )

        second_model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer
        )

        first_callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=100,
                mode='auto'
            ),
            F1((x_train, y_train)),
        ]

        first_model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=1000,
            validation_data=(x_val, y_val),
            callbacks=first_callbacks
        )

        print("Unfreezing embeddings...")

        weights = first_model.get_weights()
        second_model.set_weights(weights)

        with temp_dir() as model_dir:

            model_path = str(model_dir / "best_model.hdf5")
            second_callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    verbose=100,
                    mode='auto'
                ),
                F1((x_train, y_train)),
                ModelCheckpoint(
                    model_path,
                    "val_f1",
                    verbose=100,
                    save_best_only=True,
                    mode="max"
                )
            ]
            second_model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=1000,
                validation_data=(x_val, y_val),
                callbacks=second_callbacks
            )
            second_model.load_weights(model_path)

        probs = second_model.predict(x_val)
        argmax = np.argmax(probs, axis=1)
        preds = np.zeros_like(probs)
        preds[np.arange(preds.shape[0]), argmax] = 1.0
        f1 = f1_score(y_val[:-1], preds[:-1], average=None).mean()

        if best_f1 is None or f1 > best_f1:
            best_model = second_model
            best_f1 = f1
            best_config = config

    return best_model, best_f1, best_config


def _train_test_split(dataset, test_ratio, random_state):
    train_dataset = deepcopy(dataset)
    train_dataset[INTENTS] = dict()
    test_dataset = deepcopy(dataset)
    test_dataset[INTENTS] = dict()

    for intent_name, intent in iteritems(dataset[INTENTS]):
        utterances = random_state.permutation(intent[UTTERANCES])
        test_ix = int(len(utterances) * test_ratio)
        train_dataset[INTENTS][intent_name] = {
            UTTERANCES: utterances[:-test_ix].tolist()}
        test_dataset[INTENTS][intent_name] = {
            UTTERANCES: utterances[-test_ix:].tolist()}

    return train_dataset, test_dataset
