import json
import operator
import os
import shutil
import spacy
from collections import defaultdict

from rasa_nlu import train
from rasa_nlu.training_data import TrainingData
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer

from custom_intent_parser.intent_parser.intent_parser import IntentParser
from custom_intent_parser.utils import LimitedSizeDict, transform_to_rasa_format

class RasaIntentParser(IntentParser):
    def __init__(self, backend="spacy_sklearn", language="en", cache_size=100):

        self.backend=backend
        self.language=language
        self.num_threads=1

        backend=self.backend
        if backend.lower()=='mitie':
            from rasa_nlu.interpreters.mitie_interpreter import MITIEInterpreter
            from rasa_nlu.trainers.mitie_trainer import MITIETrainer
            self.mitie_file="./data/total_word_feature_extractor.dat"
            self.interpreter=MITIEInterpreter()
            self.trainer=MITIETrainer(self.mitie_file, self.language, self.num_threads)
        elif backend.lower() == 'spacy_sklearn':
            from rasa_nlu.interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
            from rasa_nlu.trainers.spacy_sklearn_trainer import SpacySklearnTrainer
            self.interpreter=SpacySklearnInterpreter()
            self.trainer=SpacySklearnTrainer({}, self.language, self.num_threads)
        else:
            raise NotImplementedError("other backend trainers not implemented yet")

        self._cache = LimitedSizeDict(size_limit=cache_size)


    def fitted(self):
        return self.didfit

    def fit(self, dataset):

        dir_name='__rasa_tmp'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        training_file_name="{0}/training_data.json".format(dir_name)
        with open(training_file_name, 'w') as outfile:
            json.dump(transform_to_rasa_format(dataset), outfile)

        training_data = TrainingData(training_file_name, self.backend, self.language)
        
        self.trainer.train(training_data)
        self.interpreter.nlp=spacy.load(self.language, parser=False, entity=False, matcher=False)
        self.interpreter.featurizer=SpacyFeaturizer(self.trainer.nlp)
        self.interpreter.classifier=self.trainer.intent_classifier
        self.interpreter.extractor=self.trainer.entity_extractor

        self.didfit=True

        return self

    def parse(self, text):
        if text not in self._cache:
            self._update_cache(text)
        return self._cache[text]

    def get_intent(self, text):
        if text not in self._cache:
            self._update_cache(text)
        parse = self._cache[text]
        return {"intent": parse["intent"], "text": text}

    def get_entities(self, text, intent=None):
        if text not in self._cache:
            self._update_cache(text)
        parse = self._cache[text]
        return {"entities": parse["entities"], "text": text}

    def _update_cache(self, text, intent=None):
        self.check_fitted()

        self.check_fitted()

        rasa_result=self.interpreter.parse(unicode(text))
        intent={
            'intent': rasa_result['intent'],
            'proba': rasa_result.get('confidence', None)
        }
        entities=[]

        for entity in rasa_result['entities']:
            value=entity["value"]
            if value not in text:
                raise ValueError("Rasa returned unknown entity: %s"%value)
            entities.append({
                "range": (entity["start"],entity["end"]),
                "value": value,
                "entity": entity["entity"]
            })

        if intent['intent'] is 'Other':
            result={
                "text": text,
                "entities": [],
                "intent": None
            }
        else:
            result={
                "text": text,
                "entities": entities,
                "intent": intent
            }
        self._cache[text] = result
        return

    def load():
        pass

    def save():
        pass
