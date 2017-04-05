import numpy as np
from sklearn.linear_model import SGDClassifier

from data_augmentation import augment_dataset, get_non_empty_intents, \
    get_regularization_factor
from feature_extraction import Featurizer
from intent_classifier import IntentClassifier
from snips_nlu.constants import LANGUAGE
from snips_nlu.languages import Language
from snips_nlu.preprocessing import verbs_stems, stem
from snips_nlu.result import IntentClassificationResult
from snips_nlu.utils import (instance_to_generic_dict, ensure_string,
                             safe_pickle_dumps, safe_pickle_loads)


def get_default_parameters():
    return {
        "loss": 'log',
        "penalty": 'l2',
        "class_weight": 'balanced',
        "n_iter": 5,
        "random_state": 42,
        "n_jobs": -1
    }


class SnipsIntentClassifier(IntentClassifier):
    def __init__(self, language, classifier_args=get_default_parameters()):
        self.language = language
        self.classifier_args = classifier_args
        self.classifier = None
        self.intent_list = None
        self.featurizer = Featurizer(self.language)

    @property
    def fitted(self):
        return self.intent_list is not None

    def fit(self, dataset):
        self.intent_list = get_non_empty_intents(dataset)

        if len(self.intent_list) > 0:
            queries, y = augment_dataset(dataset, self.language,
                                         self.intent_list)
            self.featurizer = self.featurizer.fit(queries, y)
            X = self.featurizer.transform(queries)
            alpha = get_regularization_factor(dataset)
            self.classifier_args.update({'alpha': alpha})
            self.classifier = SGDClassifier(**self.classifier_args).fit(X, y)
            self.intent_list = [None] + self.intent_list
        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError('SnipsIntentClassifier instance must be '
                                 'fitted before `get_intent` can be called')

        if len(text) == 0 or len(self.intent_list) == 0:
            return None

        stemmed_text = stem(text, self.language)

        X = self.featurizer.transform([stemmed_text])
        proba_vect = self.classifier.predict_proba(X)
        predicted = np.argmax(proba_vect[0])

        intent_name = self.intent_list[int(predicted)]
        prob = proba_vect[0][int(predicted)]

        if intent_name is None:
            return None

        return IntentClassificationResult(intent_name, prob)

    def to_dict(self):
        obj_dict = instance_to_generic_dict(self)
        obj_dict.update({
            "classifier_args": self.classifier_args,
            "classifier_pkl": safe_pickle_dumps(self.classifier),
            "intent_list": self.intent_list,
            "language_code": self.language.iso_code,
            "featurizer": self.featurizer.to_dict()
        })
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        language = Language.from_iso_code(obj_dict['language_code'])
        classifier = cls(language=language,
                         classifier_args=obj_dict['classifier_args'])
        obj_dict['classifier_pkl'] = ensure_string(obj_dict['classifier_pkl'])
        classifier.classifier = safe_pickle_loads(obj_dict['classifier_pkl'])
        classifier.intent_list = obj_dict['intent_list']
        classifier.featurizer = Featurizer.from_dict(obj_dict['featurizer'])
        return classifier
