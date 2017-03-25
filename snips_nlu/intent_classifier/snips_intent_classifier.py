import numpy as np
import random
import json

from intent_classifier import IntentClassifier
from feature_extraction import Featurizer
from data_augmentation  import augment_dataset
from snips_nlu.result import IntentClassificationResult

from sklearn.linear_model import SGDClassifier
from sklearn.utils.validation import check_is_fitted

class SnipsIntentClassifier(IntentClassifier):

    def __init__(self):
        
        self.language = 'en'
        self.best_feat = None
        self.clf = None
        self.intent_list = None
        self.featurizer = Featurizer()

    @property
    def fitted(self):
        return check_is_fitted(self.clf, "t_")

    def fit(self, dataset):

        (queries, y), alpha, self.intent_list = augment_dataset(dataset, self.language)

        X = self.featurizer.fit_transform(queries, y)
        
        clf = SGDClassifier(loss='log', penalty='l2', alpha=alpha, class_weight = 'balanced', n_iter=5, random_state=42, n_jobs=-1)
        self.clf = clf.fit(X, y)

        return self

    def get_intent(self, text):

        X = self.featurizer.transform([text])

        proba_vect = self.clf.predict_proba(X)
        predicted = np.argmax(proba_vect[0])
        
        intent_name = self.intent_list[int(predicted)]
        prob = proba_vect[0][int(predicted)]

        return IntentClassificationResult(intent_name, prob)

