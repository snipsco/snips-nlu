import numpy as np
from sklearn.linear_model import SGDClassifier

from data_augmentation import augment_dataset, get_non_empty_intents
from feature_extraction import Featurizer
from intent_classifier import IntentClassifier
from snips_nlu.result import IntentClassificationResult


class SnipsIntentClassifier(IntentClassifier):
    def __init__(self):
        self.language = 'en'
        self.clf = None
        self.intent_list = None
        self.featurizer = Featurizer(language=self.language)

    @property
    def fitted(self):
        return self.intent_list is not None

    @property
    def is_empty(self):
        return len(self.intent_list) == 0

    def fit(self, dataset):
        self.intent_list = get_non_empty_intents(dataset)

        if not self.is_empty:
            (queries, y), alpha = augment_dataset(dataset, self.intent_list, self.language)
            X = self.featurizer.fit_transform(queries, y)
            clf = SGDClassifier(loss='log', penalty='l2', alpha=alpha, class_weight='balanced', n_iter=5,
                                random_state=42, n_jobs=-1)
            self.clf = clf.fit(X, y)

        return self

    def get_intent(self, text):
        if not self.fitted:
            raise AssertionError('Fit before predict.')

        if len(text) == 0:
            print 'empty query !'
            return None

        if self.is_empty:
            print 'empty training set !'
            return None

        self.intent_list = ['None'] + self.intent_list

        X = self.featurizer.transform([text])
        proba_vect = self.clf.predict_proba(X)
        predicted = np.argmax(proba_vect[0])

        intent_name = self.intent_list[int(predicted)]
        prob = proba_vect[0][int(predicted)]

        return IntentClassificationResult(intent_name, prob)
