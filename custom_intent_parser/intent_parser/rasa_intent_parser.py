import io
import json
import os
import pickle
import shutil
import spacy

from rasa_nlu.training_data import TrainingData
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from custom_intent_parser.intent_parser.intent_parser import IntentParser
from custom_intent_parser.result import (intent_classification_result,
                                         parsed_entity, result)
from custom_intent_parser.utils import LimitedSizeDict
from custom_intent_parser.rasa_utils import transform_to_rasa_format


class RasaIntentParser(IntentParser):

    def __init__(self, backend="spacy_sklearn", language="en", cache=None,
                 cache_size=100):

        self.backend = backend
        self.language = language
        self.num_threads = 1
        self.train_dataset = None

        if self.backend == 'spacy_sklearn':
            from rasa_nlu.interpreters.spacy_sklearn_interpreter import (
                SpacySklearnInterpreter)
            from rasa_nlu.trainers.spacy_sklearn_trainer import (
                SpacySklearnTrainer)
            self.interpreter = SpacySklearnInterpreter()
            self.trainer = SpacySklearnTrainer(
                {}, self.language, self.num_threads)
            self.interpreter.nlp = spacy.load(
                self.language, parser=False, entity=False, matcher=False)
        else:
            supported_backends = ['spacy_sklearn']
            raise NotImplementedError("%s not supported. Supported backends ar\
                e %s" % (self.backend, supported_backends))

        if cache is None:
            cache = LimitedSizeDict(size_limit=cache_size)
        self._cache = cache

    def fitted(self):
        return (self.train_dataset is not None)

    def fit(self, dataset):

        if len(dataset.queries.keys()) < 2:
            raise ValueError("Rasa backend can only be used on a dataset \
                containing multiple intents")

        dir_name = '__rasa_tmp'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        training_file_name = os.path.join(dir_name, "training_data.json")
        with io.open(training_file_name, 'w') as f:
            f.write(unicode(json.dumps(transform_to_rasa_format(dataset),
                                       ensure_ascii=False)))

        training_data = TrainingData(
            training_file_name, self.backend, self.language)

        shutil.rmtree(dir_name)

        self.train_dataset = dataset
        self.trainer.train(training_data)
        self.interpreter.featurizer = SpacyFeaturizer(self.trainer.nlp)
        self.interpreter.classifier = self.trainer.intent_classifier
        self.interpreter.extractor = self.trainer.entity_extractor

        return self

    def parse(self, text):
        if text not in self._cache:
            self._update_cache(text)
        return self._cache[text]

    def get_intent(self, text):
        if text not in self._cache:
            self._update_cache(text)
        parse = self._cache[text]

        return intent_classification_result(
            intent_name=parse["intent"]["intent"],
            prob=parse["intent"])

    def get_entities(self, text, intent=None):
        if text not in self._cache:
            self._update_cache(text)
        parse = self._cache[text]
        return parse["entities"]

    def _update_cache(self, text, intent=None):
        self.check_fitted()

        rasa_result = self.interpreter.parse(unicode(text))
        intent = intent_classification_result(intent_name=rasa_result['intent'],
                                              prob=rasa_result.get('confidence',
                                                                   None))
        entities = []

        for entity in rasa_result['entities']:
            value = entity["value"]
            if value not in text:
                raise ValueError("Rasa returned unknown entity: %s" % value)
            entities.append(parsed_entity(
                (entity["start"], entity["end"]),
                value,
                entity["entity"]))

        r = result(text, parsed_intent=intent,
                       parsed_entities=entities)
        self._cache[text] = r
        return

    @staticmethod
    def train_dataset_file_name(path):
        return os.path.join(path, "train_dataset.json")

    @staticmethod
    def intent_parser_file_name(path):
        return os.path.join(path, "intent_parser.json")

    @classmethod
    def load(cls, path):
        with io.open(cls.intent_parser_file_name(path), encoding="utf8") as f:
            data = json.load(f)

        with io.open(cls.train_dataset_file_name(path), "rb") as f:
            train_dataset = pickle.load(f)

        backend = data["backend"]
        language = data["language"]
        cache_size = data["cache_size"]
        cache = LimitedSizeDict([(k, v) for k, v in data["cache_items"]],
                                size_limit=cache_size)

        parser = cls(backend, language, cache, cache_size)
        parser.fit(train_dataset)

        return parser

    def save(self, path):
        self_as_dict = dict()
        self_as_dict["cache_size"] = self._cache.size_limit
        self_as_dict["cache_items"] = self._cache.items()
        self_as_dict["backend"] = self.backend
        self_as_dict["language"] = self.language
        os.mkdir(path)

        self_as_dict["train_dataset_file_name"] = \
            self.train_dataset_file_name(path)

        with io.open(self_as_dict["train_dataset_file_name"], "wb") as f:
            pickle.dump(self.train_dataset, f)

        with io.open(self.intent_parser_file_name(path), "w",
                     encoding="utf8") as f:
            data = json.dumps(self_as_dict, indent=2)
            f.write(unicode(data))
