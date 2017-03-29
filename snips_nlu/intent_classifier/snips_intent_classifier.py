import cPickle

import numpy as np
from sklearn.linear_model import SGDClassifier

from data_augmentation import augment_dataset, get_non_empty_intents
from feature_extraction import Featurizer
from intent_classifier import IntentClassifier
from snips_nlu.result import IntentClassificationResult
from snips_nlu.utils import instance_to_generic_dict


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
    def __init__(self, classifier_args=get_default_parameters(),
                 classifier=None, intent_list=None, featurizer=None):
        self.classifier_args = classifier_args
        self.classifier = classifier
        self.intent_list = intent_list
        self.featurizer = featurizer

    @property
    def fitted(self):
        return self.intent_list is not None

    def fit(self, dataset):
        if self.featurizer is None:
            self.featurizer = Featurizer(dataset["language"])

        self.intent_list = get_non_empty_intents(dataset)

        if len(self.intent_list) > 0:
            (queries, y), alpha = augment_dataset(dataset, self.intent_list)
            X = self.featurizer.fit_transform(queries, y)
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

        X = self.featurizer.transform([text])
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
            "classifier_pkl": cPickle.dumps(self.classifier),
            "intent_list": self.intent_list,
            "featurizer": self.featurizer.to_dict()
        })
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(
            classifier_args=obj_dict['classifier_args'],
            classifier=cPickle.loads(obj_dict['classifier_pkl']),
            intent_list=obj_dict['intent_list'],
            featurizer=Featurizer.from_dict(obj_dict['featurizer'])
        )
