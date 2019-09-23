from __future__ import division

import json
import logging
import time
from abc import ABCMeta, abstractmethod
from builtins import object, range, zip
from copy import deepcopy
from pathlib import Path

import numpy as np
import requests
import torch
from dataclasses import dataclass
from future.utils import viewvalues, with_metaclass
from pytorch_transformers import AdamW, BertModel, BertTokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import CosineSimilarity as TorchCosineSimilarity, Embedding, LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Subset,
    TensorDataset
)
from tqdm import tqdm

from snips_nlu.common.registrable import Registrable
from snips_nlu.common.utils import (
    check_persisted_path, check_random_state, json_string)
from snips_nlu.constants import (BERT_MODEL_PATH, DATA, INTENTS, LANGUAGE,
                                 RES_PROBA, TEXT, UTTERANCES)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.exceptions import LoadingError, _EmptyDatasetUtterancesError
from snips_nlu.intent_classifier import (
    Featurizer as LinguisticFeaturizer,
    IntentClassifier,
    LogRegIntentClassifier)
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    build_training_data, text_to_utterance)
from snips_nlu.pipeline.configs.intent_classifier import (
    LogRegIntentClassifierWithParaphraseConfig, ParaphraseClassifierConfig)
from snips_nlu.result import intent_classification_result

logger = logging.getLogger(__name__)

PARAPHRASE_SERVICE_URL = "http://localhost:8000/api/1.0"
PIVOTS = {
    "en": ["de", "fr", "ru", "es", "it"]
}


def _utterance_text(u):
    return "".join(c[TEXT] for c in u[DATA])


def get_f1(num_labels):
    def f1(y_true, y_pred):
        return f1_score(
            y_pred, y_true, average="macro", labels=range(num_labels))

    return f1


@IntentClassifier.register("log_reg_intent_classifier_with_paraphrase")
class LogRegIntentClassifierWithParaphrase(LogRegIntentClassifier):
    config_type = LogRegIntentClassifierWithParaphraseConfig

    def __init__(self, config=None, **shared):
        super(LogRegIntentClassifierWithParaphrase, self).__init__(
            config=config, **shared)
        self.config.data_augmentation_config.min_utterances = 0
        self.config.data_augmentation_config.unknown_word_prob = None
        self.config.data_augmentation_config \
            .add_builtin_entities_examples = False
        self.log_dir = shared["log_dir"]
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
        self.output_dir = shared["output_dir"]
        if not self.output_dir.parent.exists():
            self.output_dir.parent.mkdir(parents=True)
        self.global_step = shared.get("global_step", 0)
        self._linguistic_featurizer = None
        self._tokenizer = None
        self._bert_model_path = None
        self._runner = None
        self.intent_list = None
        self.language = None

    @property
    def fitted(self):
        return self._runner is not None

    def fit(self, dataset):
        logger.info("Fitting LogRegIntentClassifier...")
        dataset = validate_and_format_dataset(dataset)
        # Remove slots to avoid dirty context/slot augmentation
        remove_slots(dataset)
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        self.language = dataset[LANGUAGE]

        random_state = check_random_state(self.random_state)

        self._bert_model_path = self.config. \
            paraphrase_classifier_config. \
            similarity_scorer_config[
            "sentence_embedder_config"][BERT_MODEL_PATH]
        self._tokenizer = BertTokenizer.from_pretrained(self._bert_model_path)

        data_augmentation_config = self.config.data_augmentation_config
        utterances, classes, intent_list = build_training_data(
            dataset, self.language, data_augmentation_config, self.resources,
            random_state)

        self.intent_list = intent_list
        if len(self.intent_list) <= 1:
            return self

        self._linguistic_featurizer = LinguisticFeaturizer(
            config=self.config.featurizer_config,
            builtin_entity_parser=self.builtin_entity_parser,
            custom_entity_parser=self.custom_entity_parser,
            resources=self.resources,
            random_state=self.random_state,
        )
        self._linguistic_featurizer.language = self.language
        self.none_class = max(classes)

        class_weights_arr = compute_class_weight(
            "balanced", range(self.none_class + 1), classes)
        # Re-weight the noise class
        class_weights_arr[-1] *= self.config.noise_reweight_factor

        # Extract linguistic features for all utterances
        text_utterances = [
            _utterance_text(utterances[i]) for i in range(len(utterances))]
        not_none_ix = [i for i, c in enumerate(classes)
                       if c != self.none_class]
        none_ix = [i for i, c in enumerate(classes)
                   if c == self.none_class]
        utterances_to_paraphrase = [text_utterances[i] for i in not_none_ix]
        num_paraphrase_to_generate = self.config.n_paraphrases - 1
        paraphrases = _get_paraphrases(
            utterances_to_paraphrase,
            PARAPHRASE_SERVICE_URL,
            self.language,
            PIVOTS[self.language],
            num_paraphrase_to_generate,
            random_state=random_state,
        )
        # Prepend original sentence to paraphrases
        for i, u in enumerate(utterances_to_paraphrase):
            paraphrases[i] = [u] + paraphrases[i]
        paraphrases = [text_to_utterance(pp) for p in paraphrases for pp in p]

        # Fit featurizer
        y = np.repeat(
            classes[not_none_ix],
            [self.config.n_paraphrases for _ in range(len(not_none_ix))]
        )
        y = np.concatenate((y, classes[none_ix]))
        try:
            self._linguistic_featurizer.fit(
                dataset,
                paraphrases + [utterances[i] for i in none_ix],
                y,
                self.none_class,
            )
        except _EmptyDatasetUtterancesError:
            logger.warning("No (non-empty) utterances found in dataset")
            self.featurizer = None
            return self

        # Create examples
        paraphrase_it = (p for p in paraphrases)
        not_none_paraphrases = [
            [next(paraphrase_it)["data"][0]["text"]
             for _ in range(self.config.n_paraphrases)]
            for _ in not_none_ix]
        none_paraphrases = [
            [text_utterances[i] for _ in range(self.config.n_paraphrases)]
            for i in none_ix]
        all_paraphrases = not_none_paraphrases + none_paraphrases
        labels = np.concatenate((classes[not_none_ix], classes[none_ix]))
        examples = self._build_examples(
            all_paraphrases,
            classes=labels,
            random_state=random_state,
        )
        clf_config = self.config.paraphrase_classifier_config
        linguistic_features_size = examples[
            0].paraphrases_linguistic_features.shape[-1]
        clf_config.n_paraphrases = self.config.n_paraphrases
        clf_config.sentence_classifier_config[
            "input_size"] = linguistic_features_size
        clf_config.sentence_classifier_config["output_size"] = len(
            set(classes))
        clf_config.class_weights = np.array(
            class_weights_arr, dtype="float32")

        clf = ParaphraseClassifier(clf_config)
        clf.none_class = self.none_class
        self._runner = Runner(
            clf,
            global_step=self.global_step,
            random_state=self.random_state,
            **self.config.runner_config
        )

        train_examples, eval_examples = train_test_split(
            examples,
            test_size=self.config.validation_ratio,
            random_state=random_state,
        )
        training_loader = _data_loader_from_examples(
            train_examples,
            self.config.batch_size,
            return_labels=True,
            shuffle=True
        )
        eval_loader = _data_loader_from_examples(
            eval_examples,
            self.config.batch_size,
            return_labels=True,
            shuffle=True
        )

        debug_dataloader = DataLoader(
            Subset(eval_loader.dataset, list(range(self.config.batch_size))),
            batch_size=self.config.batch_size,
        )

        debug_config = {
            "loader": debug_dataloader,
            "labels": intent_list,
            "examples": eval_examples[:self.config.batch_size]
        }

        writer = SummaryWriter(log_dir=str(self.log_dir))
        self._runner.model.none_cls = self.none_class

        # Train the network
        optimized_params = self._runner.model.pretrainable_params
        optimized_params = [
            {
                "params": [p for n, p in optimized_params],
                "weight_decay": 0.0,
                "names": [n for n, p in optimized_params]
            }
        ]
        optimizer = AdamW(optimized_params, **self.config.optimizer_config)
        optimized_param_names = [n for param_set in optimizer.param_groups
                                 for n in param_set["names"]]
        logger.debug(
            f"Optimizing pretrainable parameters: {optimized_param_names}")
        # Pre-train classifier layer
        self._runner.train(
            training_loader,
            eval_loader,
            get_f1(len(intent_list)),
            optimizer,
            self.config.n_epochs,
            output_dir=self.output_dir,
            writer=writer,
            debug_config=debug_config,
        )

        # Fine-tune the full network if needed
        finetunable_params = self._runner.model.finetunable_params
        if finetunable_params:
            no_decay = {"bias", "LayerNorm.weight"}
            finetunable_params = [
                {
                    "params": [p for n, p in finetunable_params
                               if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                    "names": [n for n, p in finetunable_params
                              if not any(nd in n for nd in no_decay)],
                },
                {
                    "params": [p for n, p in finetunable_params
                               if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "names": [n for n, p in finetunable_params
                              if any(nd in n for nd in no_decay)],
                }
            ]
            for g in finetunable_params:
                optimizer.add_param_group(g)
            optimized_param_names = [n for param_set in optimizer.param_groups
                                     for n in param_set["names"]]
            logger.debug(f"Optimized all parameters: {optimized_param_names}")
            self._runner.train(
                training_loader,
                eval_loader,
                get_f1(len(intent_list)),
                optimizer,
                self.config.n_epochs,
                output_dir=self.output_dir,
                writer=writer,
                debug_config=debug_config,
            )

        self.global_step = self._runner.global_step
        return self

    @check_persisted_path
    def persist(self, path):
        path.mkdir()

        featurizer = None
        if self._linguistic_featurizer is not None:
            featurizer = "featurizer"
            featurizer_path = path / featurizer
            self._linguistic_featurizer.persist(featurizer_path)

        runner = None
        if self._runner is not None:
            runner = "runner"
            self._runner.persist(path / runner)

        self_as_dict = {
            "config": self.config.to_dict(),
            "intent_list": self.intent_list,
            "featurizer": featurizer,
            "runner": runner,
            "global_step": self.global_step,
            "language": self.language,
        }
        if self._tokenizer is not None:
            self_as_dict["tokenizer"] = self._bert_model_path

        classifier_json = json_string(self_as_dict)
        with (path / "intent_classifier.json").open(
                mode="w", encoding="utf8") as f:
            f.write(classifier_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)
        model_path = path / "intent_classifier.json"
        if not model_path.exists():
            raise LoadingError("Missing intent classifier model file: %s"
                               % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model_dict = json.load(f)

        # Create the classifier
        config = LogRegIntentClassifierWithParaphraseConfig.from_dict(
            model_dict["config"])

        self = cls(config, **shared)

        if model_dict["intent_list"] is not None:
            self.intent_list = model_dict["intent_list"]

        self.language = model_dict["language"]

        # Create the underlying ParaphraseClassifier
        runner = model_dict.get("runner")
        if runner is not None:
            runner = Runner.from_path(path / runner)
        self._runner = runner

        # Add the featurizer
        featurizer = model_dict["featurizer"]
        if featurizer is not None:
            featurizer_path = path / featurizer
            self._linguistic_featurizer = LinguisticFeaturizer.from_path(
                featurizer_path, **shared)

        # Add the tokenizer
        tokenizer = model_dict.get("tokenizer")
        if tokenizer:
            self._tokenizer = BertTokenizer.from_pretrained(tokenizer)

        return self

    def _get_intents(self, text, intents_filter):
        return self.batch_get_intent([text], intents_filter=intents_filter)

    def batch_get_intents(self, texts, intents_filter=None):
        if isinstance(intents_filter, str):
            intents_filter = {intents_filter}
        elif isinstance(intents_filter, list):
            intents_filter = set(intents_filter)

        if not texts or not self.intent_list \
                or not self._linguistic_featurizer:
            results = [intent_classification_result(None, 1.0)]
            results += [intent_classification_result(i, 0.0)
                        for i in self.intent_list if i is not None]
            return results * len(texts)

        if len(self.intent_list) == 1:
            return [intent_classification_result(self.intent_list[0], 1.0)
                    ] * len(texts)

        # Set the random state to avoid stochastic inference
        random_state = check_random_state(0)
        paraphrases = _get_paraphrases(
            texts,
            PARAPHRASE_SERVICE_URL,
            self.language,
            PIVOTS[self.language],
            self.config.n_paraphrases - 1,
            random_state=random_state,
        )
        # Prepend original sentence to paraphrases
        for i, t in enumerate(texts):
            paraphrases[i] = [t] + paraphrases[i]
        examples = self._build_examples(paraphrases, random_state)
        loader = _data_loader_from_examples(
            examples, batch_size=self.config.batch_size, shuffle=False)
        proba_vec = self._runner.predict(loader)[0].tolist()
        if intents_filter is not None:
            intents_filter = set(intents_filter)
        results = []
        for probs in proba_vec:
            res = [
                intent_classification_result(i, proba)
                for i, proba in zip(self.intent_list, probs)
                if intents_filter is None or i is None or i in intents_filter
            ]
            results.append(res)
        return [sorted(res_, key=lambda res: -res[RES_PROBA])
                for res_ in results]

    def batch_get_intent(self, texts, intents_filter=None):
        return [res[0] for res in
                self.batch_get_intents(texts, intents_filter=intents_filter)]

    def _build_examples(self, paraphrases, random_state, classes=None):
        # Extract linguistic features for all utterances
        try:
            x = self._linguistic_featurizer.transform(
                [text_to_utterance(p) for pp in paraphrases for p in pp])
        except _EmptyDatasetUtterancesError:
            logger.warning("No (non-empty) utterances found in dataset")
            self._linguistic_featurizer = None
            return self
        x = np.asarray(x.todense()).reshape(
            (len(paraphrases), self.config.n_paraphrases, -1))
        examples = _create_examples(
            paraphrases,
            x,
            self._tokenizer,
            labels=classes,
            random_state=random_state,
        )
        return examples


class ParaphraseClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_paraphrases = config.n_paraphrases or 1
        self.similarity_scorer = SimilarityScorer(
            config.similarity_scorer_config)

        sentence_classifier_config = deepcopy(
            config.sentence_classifier_config)
        sentence_classifier_name = sentence_classifier_config.pop("name")
        sentence_classifier_cls = SentenceClassifier.by_name(
            sentence_classifier_name)
        self.sentence_classifier = sentence_classifier_cls(
            sentence_classifier_config)

        weight = config.class_weights
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self._loss = nn.NLLLoss(weight=weight)
        # Init weights only for the classifier, and keep the similarity scorer
        # pretrained
        self.sentence_classifier.apply(init_weights)
        self._none_cls = None
        self._pretrainable_params = (
                self.sentence_classifier.pretrainable_params +
                self.similarity_scorer.pretrainable_params)
        self._finetunable_params = (
                self.sentence_classifier.finetunable_params +
                self.similarity_scorer.finetunable_params)

    @property
    def none_class(self):
        return self._none_cls

    @none_class.setter
    def none_class(self, value):
        self._none_cls = value

    @property
    def pretrainable_params(self):
        return self._pretrainable_params

    @property
    def finetunable_params(self):
        return self._finetunable_params

    def forward(
            self,
            paraphrases_input_ids,  # (bsz, n_paraphrases, max_length)
            paraphrases_masks,  # (bsz, n_paraphrases, max_length)
            linguistic_features,  # (bsz, n_paraphrases, n_features)
            y=None,  # (bsz)
            class_weights=None,
    ):
        batch_size = linguistic_features.shape[0]
        n_paraphrases = linguistic_features.shape[1]
        n_linguistic_features = linguistic_features.shape[2]

        true = torch.ones((batch_size,), dtype=torch.uint8)
        false = torch.zeros((batch_size,), dtype=torch.uint8)
        none_index = false
        not_none_index = true
        if y is not None:
            if self.none_class is None:
                raise ValueError("None class should be set before training")
            none_index = torch.where(y == self.none_cls, true, false)
            not_none_index = torch.where(none_index == false, true, false)
        none_index = none_index.bool()
        not_none_index = not_none_index.bool()

        # Classify all paraphrases
        # (bsz, n_outputs)
        linguistic_features_ = linguistic_features[not_none_index].reshape(
            -1, n_linguistic_features)
        num_none = 0
        if y is not None:
            none_linguistic_features = linguistic_features[none_index, 0]
            num_none = none_linguistic_features.shape[0]
            linguistic_features_ = torch.cat(
                (linguistic_features_, none_linguistic_features))

        # (bsz * n_paraphrases, n_classes)
        probs = self.sentence_classifier(linguistic_features_)
        unweighted_probs = torch.zeros(
            (batch_size, n_paraphrases, probs.shape[-1]))

        # Reorder probabilities
        if num_none:
            unweighted_probs[not_none_index] = probs[:-num_none].reshape(
                (batch_size - num_none, n_paraphrases, probs.shape[-1]))
            unweighted_probs[none_index, :] = probs[-num_none:].unsqueeze(1)
        else:
            unweighted_probs = probs.reshape((batch_size, n_paraphrases, -1))

        #######################
        # Compute similarity only not none queries for none queries,
        # all paraphrase have the same weight since we only consider the
        # original sentence
        # (bsz, n_paraphrases)
        similarities = torch.ones((batch_size, n_paraphrases))
        if batch_size - num_none:
            similarities[not_none_index] = self.similarity_scorer(
                paraphrases_input_ids[not_none_index],
                paraphrases_masks[not_none_index],
            )
        similarities = torch.softmax(similarities, dim=-1)
        weighted_probs = torch.sum(
            unweighted_probs * similarities.unsqueeze(2),
            dim=1
        )
        #######################
        # To mimic the log reg uncomment this section and set n_parapharse to 1
        # similarities = torch.ones((batch_size, n_paraphrases))
        # weighted_probs = unweighted_probs[:, 0]
        #######################

        # Predict
        if y is None:
            return weighted_probs, unweighted_probs, similarities

        # Compute loss
        loss = self._loss(torch.log(weighted_probs), y)
        return loss, weighted_probs, unweighted_probs, similarities

    @check_persisted_path
    def persist(self, path):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        state = "state.pt"
        torch.save(self.state_dict(), str(path / state))
        config = self.config.to_dict()
        config["class_weights"] = None
        if self.config.class_weights is not None:
            config["class_weights"] = self.config.class_weights.tolist()
        self_as_dict = {
            "state": state,
            "config": config
        }
        with (path / "classifier.json").open("w") as f:
            json.dump(self_as_dict, f)

    @classmethod
    def from_path(cls, path):
        path = Path(path)
        with (path / "classifier.json").open() as f:
            clf = json.load(f)
        config = ParaphraseClassifierConfig.from_dict(clf["config"])
        self = cls(config)
        if clf["state"] is not None:
            self.load_state_dict(torch.load(str(path / clf["state"])))
        return self


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class SentenceClassifier(with_metaclass(ABCMeta, nn.Module, Registrable)):
    pass


@SentenceClassifier.register("mlp_intent_classifier")
class MLPSentenceClassifier(SentenceClassifier):
    def __init__(self, config):
        super(MLPSentenceClassifier, self).__init__()
        hidden_sizes = [config["input_size"]] + config["hidden_sizes"]
        hidden_sizes += [config["output_size"]]
        hiddens = [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
                   for i in range(len(hidden_sizes))[:-1]]
        activation_class = getattr(nn, config["activation"])
        activations = [activation_class()
                       for _ in config["hidden_sizes"]]
        activations.append(nn.Softmax(dim=-1))
        dropouts = [None for _ in hiddens]
        if config.get("dropout"):
            dropouts = [nn.Dropout(config["dropout"]) for _ in hiddens[:-1]]
            dropouts.append(None)
        layers = []
        for i, (hidden, activation, dropout) in enumerate(
                zip(hiddens, activations, dropouts)):
            layers.extend([hidden, activation])
            if dropout:
                layers.append(dropout)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x)
        return output

    @property
    def pretrainable_params(self):
        return list(self.named_parameters())

    @property
    def finetunable_params(self):
        return []


@SentenceClassifier.register("cnn_intent_classifier")
class CNNSentenceClassifier(SentenceClassifier):
    pass


@SentenceClassifier.register("lstm_intent_classifier")
class LSTMSentenceClassifier(SentenceClassifier):
    pass


class SentenceEmbedder(with_metaclass(ABCMeta, nn.Module, Registrable)):
    @property
    @abstractmethod
    def embedding_size(self):
        pass

    @property
    def pretrainable_params(self):
        return list(self.named_parameters())

    @property
    def finetunable_params(self):
        return []


@SentenceEmbedder.register("bert_sentence_embedder")
class BertSentenceEmbedder(SentenceEmbedder):
    def __init__(self, config):
        super(BertSentenceEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config[BERT_MODEL_PATH])

    def forward(self, input_ids, attention_mask):
        _, outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    @property
    def embedding_size(self):
        return self.bert.config.hidden_size

    @property
    def pretrainable_params(self):
        return []

    @property
    def finetunable_params(self):
        return list(self.bert.named_parameters())


@SentenceEmbedder.register("bilstm_sentence_embedder")
class BiLSTMSentenceEmbedder(SentenceEmbedder):
    def __init__(self, config):
        super(BiLSTMSentenceEmbedder, self).__init__()
        bert = BertModel.from_pretrained(
            config[BERT_MODEL_PATH])
        bert_config = deepcopy(bert.config)
        # We only keep the BERT word embedding, not the position nor the token
        # type embedding
        self.embeddings = Embedding(
            bert_config.vocab_size,
            bert_config.hidden_size,
            padding_idx=0,
        )
        self.embeddings.load_state_dict(
            bert.embeddings.word_embeddings.state_dict())
        del bert
        self.lstm = LSTM(
            input_size=self.embeddings.embedding_dim,
            hidden_size=config["hidden_size"],
            num_layers=config.get("num_layers", 1),
            bidirectional=True,
            dropout=config.get("dropout", 0),
        )

    def forward(self, input_ids, attention_mask, hidden=None):
        bert_embedding = self.embeddings(input_ids)
        seq_lengths = attention_mask.sum(dim=-1)
        packed = pack_padded_sequence(
            bert_embedding,
            seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed, hidden)
        # Sum the forward and backward direction + the output of all the layers
        return h_n.sum(0)

    @property
    def embedding_size(self):
        return self.lstm.hidden_size

    @property
    def pretrainable_params(self):
        return list(self.lstm.named_parameters())

    @property
    def finetunable_params(self):
        return list(self.embeddings.named_parameters())


class Similarity(with_metaclass(ABCMeta, nn.Module, Registrable)):
    @property
    def pretrainable_params(self):
        return []

    @property
    def finetunable_params(self):
        return []


@Similarity.register("cosine_similarity")
class CosineSimilarity(Similarity):
    def __init__(self, _):
        super(CosineSimilarity, self).__init__()
        self.cosine_similarity = TorchCosineSimilarity(dim=-1)

    def forward(self, a, b):
        return self.cosine_similarity(a, b)


@Similarity.register("feedforward_similarity")
class FeedforwardSimilarity(Similarity):
    def __init__(self, config):
        super(FeedforwardSimilarity, self).__init__()
        self.linear = nn.Linear(3 * config["embedding_size"], 1)
        self.linear.apply(init_weights)

    def forward(self, a, b):
        features = torch.cat((a, b, a * b), dim=-1)
        # (bzs * n_paraphrases)
        similarities = self.linear(
            features.reshape((-1, features.shape[-1]))).squeeze()
        return similarities

    @property
    def pretrainable_params(self):
        return list(self.linear.named_parameters())


class SimilarityScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedder_config = deepcopy(config["sentence_embedder_config"])
        embedder_name = embedder_config.pop("name")
        embedder_cls = SentenceEmbedder.by_name(embedder_name)
        self.sentence_embedder = embedder_cls(embedder_config)

        embedding_size = self.sentence_embedder.embedding_size
        similarity_args = deepcopy(config["similarity_config"])
        similarity_name = similarity_args.pop("name")
        similarity_args["embedding_size"] = embedding_size
        self.similarity = Similarity.by_name(similarity_name)(similarity_args)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, token_ids, mask):  # (bsz, n_paraphrases, max_length)
        n_paraphrases = token_ids.shape[1]
        flattened_inputs = token_ids.reshape((-1, token_ids.shape[-1]))
        flattened_masks = mask.reshape((-1, mask.shape[-1]))

        embedded = self.sentence_embedder(
            input_ids=flattened_inputs, attention_mask=flattened_masks)

        # (bzs * n_paraphrases, embedding_size)
        embedded = embedded.reshape((-1, n_paraphrases, embedded.shape[-1]))

        # (bzs * n_paraphrases, embedding_size)
        original_embedded = embedded[:, 0].unsqueeze(1).expand_as(embedded)

        similarities = self.similarity(embedded, original_embedded)

        # (bzs, n_paraphrases)
        similarities = similarities.reshape((-1, n_paraphrases))
        return similarities

    @property
    def pretrainable_params(self):
        return self.sentence_embedder.pretrainable_params \
               + self.similarity.pretrainable_params

    @property
    def finetunable_params(self):
        return self.sentence_embedder.finetunable_params \
               + self.similarity.finetunable_params


def _format_loss_msg(train_loss, eval_loss, eval_criteria, epoch, max_epoch):
    return f"[{epoch}/{max_epoch}] Train loss: {train_loss:.5f}" \
           f" Evaluation loss: {eval_loss:.5f}" \
           f" Evaluation criteria: {eval_criteria:.5f}"


class Runner(object):
    def __init__(self, model, global_step=0, local_rank=-1, no_cuda=False,
                 random_state=None):
        self.model = model
        self.global_step = global_step
        self._local_rank = local_rank
        self._no_cuda = no_cuda
        self.random_state = random_state
        if self._local_rank == -1 or self._no_cuda:
            device_name = "cuda" if torch.cuda.is_available() \
                                    and not self._no_cuda else "cpu"
            self._device = torch.device(device_name)
            self._n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self._local_rank)
            self._device = torch.device("cuda", self._local_rank)
            self._n_gpu = 1
            torch.distributed.init_process_group(backend="nccl")

        if self._local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from "
                    "https://www.github.com/nvidia/apex "
                    "to use distributed and fp16 training.")
            self.model = DDP(model)
        elif self._n_gpu > 1:
            self.model = torch.nn.DataParallel(model)

    def train(self, training_loader, eval_loader, eval_fn, optimizer,
              n_epochs, output_dir, patience=10, writer=None,
              debug_config=None, eval_frequency=10):
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(100000000)
        torch.manual_seed(seed)
        if self._n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

        self.model.to(self._device)
        if not output_dir.parent.exists():
            output_dir.parent.mkdir(parents=True)

        output_dir = str(output_dir)
        train_losses = []
        eval_losses = []
        eval_criterion = []
        early_stopper = EarlyStopping(patience=patience)
        best_criteria = None

        debug_loader, debug_examples, labels = None, None, None
        if debug_config is not None:
            labels = debug_config["labels"]
            debug_examples = debug_config["examples"]
            debug_loader = debug_config["loader"]

        for epoch in range(n_epochs):
            self.model.train()  # prep model for training
            for i, batch in enumerate(training_loader):
                batch = tuple(t.to(self._device) for t in batch)
                optimizer.zero_grad()
                loss, _, _, _ = self.model(*batch)
                loss.backward()
                optimizer.step()
                loss = loss.item()
                train_losses.append(loss)
                self.global_step += batch[0].shape[0]

            if not epoch % eval_frequency:
                eval_res = self._do_eval(eval_loader, eval_fn)
                eval_loss = eval_res[0]
                eval_criteria = eval_res[1]
                eval_losses.append(eval_loss)
                eval_criterion.append(eval_criteria)

                if debug_loader:
                    self._perform_debug(
                        debug_examples,
                        debug_loader,
                        labels,
                        writer=writer
                    )

                if writer is not None:
                    writer.add_scalars(
                        "loss",
                        {"eval": eval_loss, "train": loss},
                        global_step=self.global_step,
                    )
                    writer.add_scalar(
                        "metrics/f1",
                        float(eval_criteria),
                        global_step=self.global_step,
                    )
                msg = _format_loss_msg(
                    loss, eval_loss, eval_criteria, epoch, n_epochs)
                logger.debug(msg)
                if best_criteria is None or eval_criteria > best_criteria:
                    torch.save(self.model.state_dict(), output_dir)
                stop = early_stopper(eval_criteria)
                if stop:
                    logger.debug("Early stopping model training !")
                    break

        self.model.load_state_dict(torch.load(output_dir))
        self.model.to(self._device)
        return self.model, train_losses, eval_losses, eval_criterion

    def predict(self, loader):
        all_original_probs = []
        all_weighted_probs = []
        all_similarities = []
        self.model.eval()
        for batch in tqdm(loader, desc="Predicting"):
            batch = tuple(batch_item.to(self._device) for batch_item in batch)
            with torch.no_grad():
                w_probs, original_probs, sims = self.model(*batch)
            all_weighted_probs.extend(w_probs.detach().cpu().numpy())
            all_original_probs.extend(original_probs.detach().cpu().numpy())
            all_similarities.extend(sims.detach().cpu().numpy())

        return (
            np.asarray(all_weighted_probs),
            np.asarray(all_original_probs),
            np.asarray(all_similarities),
        )

    def _do_eval(self, data_loader, eval_fn):
        eval_losses = []
        all_weighted_probs = []
        all_original_probs = []
        all_similarities = []
        labels = []
        self.model.eval()
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = tuple(batch_item.to(self._device) for batch_item in batch)
            with torch.no_grad():
                loss, w_probs, original_probs, sims = self.model(*batch)
            labels.extend(batch[-1].detach().cpu().numpy())
            eval_losses.append(loss.detach().cpu().numpy())
            all_weighted_probs.extend(w_probs.detach().cpu().numpy())
            all_original_probs.extend(original_probs.detach().cpu().numpy())
            all_similarities.extend(sims.detach().cpu().numpy())
        preds = np.argmax(np.asarray(all_weighted_probs), axis=-1)
        criterion = eval_fn(labels, preds)
        eval_loss = np.mean(eval_losses)

        return (
            eval_loss,
            criterion,
            all_weighted_probs,
            all_original_probs,
            all_similarities,
        )

    def _perform_debug(self, examples, loader, labels, writer=None):
        self.model.eval()
        weighted_probs = []
        original_probs = []
        similarities = []
        for batch in loader:
            batch = tuple(batch_item.to(self._device) for batch_item in batch)
            with torch.no_grad():
                batch = tuple(
                    batch_item.to(self._device) for batch_item in batch)
                with torch.no_grad():
                    w_probs, o_probs, sims = self.model(*batch[:-1])
                weighted_probs.extend(w_probs.detach().cpu().tolist())
                original_probs.extend(o_probs.detach().cpu().tolist())
                similarities.extend(sims.detach().cpu().tolist())
        debug_str = "# Debugging on examples"
        for i, (ex, w_probs, o_probs, sims) in enumerate(zip(
                examples, weighted_probs, original_probs, similarities)):
            i += 1
            header = f"\n## {i}/{len(examples)} {labels[ex.label]}:" \
                     f" '{ex.paraphrases_texts[0]}'"
            prob_table = _intent_table(w_probs, labels)
            para_table = _paraphrase_table(
                ex.paraphrases_texts, o_probs, sims, labels)
            debug_str += "\n" + "\n\n\n".join((header, prob_table, para_table))

        logger.debug(debug_str)
        if writer:
            writer.add_text(
                "debug/output", debug_str, global_step=self.global_step)

    def persist(self, path):
        path = Path(path)
        model = "model"
        self.model.persist(path / model)
        self_as_dict = {
            "model": model,
            "global_step": self.global_step,
            "local_rank": self._local_rank,
            "no_cuda": self._no_cuda,
        }
        with (path / "runner.json").open("w") as f:
            json.dump(self_as_dict, f)

    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)
        with (path / "runner.json").open() as f:
            runner_as_dict = json.load(f)
        model = runner_as_dict.pop("model")
        model = ParaphraseClassifier.from_path(path / model)
        return cls(
            model,
            random_state=shared.get("random_state"),
            **runner_as_dict,
        )


def _intent_table(probs, intents):
    header = _row(intents)
    table_style = _row(":---" for _ in intents)
    rows = [header, table_style, _row(probs)]
    return "\n".join(rows)


def _paraphrase_table(paraphrases, probs, similarities, intents):
    header_cols = ["Paraphrase", "Similarity"] + intents
    header = _row(header_cols)
    table_style = _row(":---" for _ in header_cols)
    rows = [header, table_style]
    for paraphrase, similarity, p in zip(paraphrases, similarities, probs):
        col = [paraphrase, similarity] + p
        row = _row(col)
        rows.append(row)
    return "\n".join(rows)


def _row(cols):
    cols = (f"{c:.2f}" if isinstance(c, float) else str(c) for c in cols)
    return "| " + " | ".join(str(c) for c in cols) + " |"


class EarlyStopping(object):

    def __init__(self, patience=10, delta=0):
        self._patience = patience
        self._delta = delta
        self._counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score <= self.best_score + self._delta:
            self._counter += 1
            logger.debug(
                f"EarlyStopping counter: {self._counter}"
                f" out of {self._patience}"
            )
            if self._counter >= self._patience:
                return True
            return False
        logger.debug(
            f"Validation criteria increased ({self.best_score:.6f} -->"
            f" {score:.6f})..."
        )
        self._counter = 0
        self.best_score = score
        return False


def _get_paraphrases(sentences, service_url, language, pivot_languages,
                     num_paraphrases, random_state):
    job = {
        "language": language,
        "texts": sentences,
        "services": ["yandex"],
        "pivot_languages": pivot_languages,
    }
    job_url = service_url + "/job"
    res = requests.post(job_url, json=job)
    assert res.status_code == 200
    job = json.loads(res.text)
    job_url = job_url + "/" + job["id"]
    sleep = 1
    max_sleep = 60
    while True:
        time.sleep(min(sleep, max_sleep))
        sleep *= 2
        res = requests.get(job_url)
        assert res.status_code == 200
        job = json.loads(res.text)
        if job["status"] == "finished":
            break
    paraphrase_url = service_url + "/paraphrase/%s" % job["id"]
    res = requests.get(paraphrase_url)
    assert res.status_code == 200
    paraphrases = json.loads(res.text)
    return [
        [pp["paraphrase"] for pp in random_state.choice(p, num_paraphrases)]
        for p in paraphrases]


@dataclass(unsafe_hash=True)
class InputExample:
    paraphrases_texts: None
    paraphrases_tokens: None
    paraphrases_input_ids: None
    paraphrases_masks: None
    paraphrases_linguistic_features: None
    label: None


def _create_examples(paraphrases, linguistic_features, tokenizer, random_state,
                     labels=None):
    examples = []
    paraphrases_tokens = [[tokenizer.tokenize(t) for t in p]
                          for p in paraphrases]
    max_length = max((
        len(toks) for p_tok in paraphrases_tokens for toks in p_tok)) + 2
    for i, (p_texts, p_toks) in enumerate(
            zip(paraphrases, paraphrases_tokens)):
        texts = []
        tokens = []
        input_ids = []
        masks = []
        for t, toks in zip(p_texts, p_toks):
            # The convention in BERT for single sequence is:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            toks = ["[CLS]"] + toks + ["[SEP]"]
            ids = tokenizer.convert_tokens_to_ids(toks)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            mask = [1] * len(ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(ids))
            ids += padding
            mask += padding

            assert len(ids) == max_length
            assert len(mask) == max_length
            texts.append(t)
            tokens.append(toks)
            input_ids.append(ids)
            masks.append(mask)

        ex = InputExample(
            paraphrases_texts=texts,
            paraphrases_tokens=tokens,
            paraphrases_input_ids=input_ids,
            paraphrases_masks=masks,
            paraphrases_linguistic_features=linguistic_features[i],
            label=None
        )
        if labels is not None:
            ex.label = labels[i]
        examples.append(ex)

    logger.debug("*** Example ***")
    for ex in random_state.choice(examples, min(len(examples), 20)):
        logger.debug("Label: %s" % ex.label)
        for j, t in enumerate(ex.paraphrases_texts):
            example_label = "Original" if j > 1 else "Paraphrase %s" % j
            logger.debug(
                f"{example_label} text: {ex.paraphrases_texts[j]}")
            logger.debug(
                f"{example_label} tokens: {ex.paraphrases_tokens[j]}")
            logger.debug(
                f"{example_label} mask: {ex.paraphrases_masks[j]}")
            logger.debug(
                f"{example_label} ids: {ex.paraphrases_input_ids[j]}")
            reverted_ids = tokenizer.convert_ids_to_tokens(
                ex.paraphrases_input_ids[j])
            logger.debug(f"{example_label} ids: {reverted_ids}\n")

    return examples


def _data_loader_from_examples(
        examples,
        batch_size,
        return_labels=False,
        local_rank=-1,
        shuffle=False
):
    input_ids = torch.tensor(
        [f.paraphrases_input_ids for f in examples], dtype=torch.long)
    input_mask = torch.tensor(
        [f.paraphrases_masks for f in examples], dtype=torch.uint8)
    linguistic_features = np.array(
        [f.paraphrases_linguistic_features for f in examples], dtype="float32")
    linguistic_features = torch.from_numpy(linguistic_features)
    tensors = [input_ids, input_mask, linguistic_features]
    if return_labels:
        labels = torch.tensor([f.label for f in examples], dtype=torch.long)
        tensors.append(labels)
    data = TensorDataset(*tensors)
    if shuffle:
        if local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
        return DataLoader(
            data,
            sampler=sampler,
            batch_size=batch_size,
        )
    return DataLoader(data, shuffle=False, batch_size=batch_size)


def remove_slots(dataset):
    for intent in viewvalues(dataset[INTENTS]):
        for u in intent[UTTERANCES]:
            text = _utterance_text(u)
            u = text_to_utterance(text)
