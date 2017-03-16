import cPickle

from sklearn_crfsuite import CRF

from snips_nlu.crf.crf_feature import CRFFeature
from snips_nlu.crf.crf_utils import add_bilou_tags, remove_bilou_tags


class CRFTokenClassifier:
    def __init__(self, model, features, use_bilou=True):
        self._model = None
        self.model = model
        self._features = None
        self.features = features
        self.use_bilou = use_bilou
        self.fitted = False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if not isinstance(value, CRF):
            raise ValueError("Expected CRF but found: %s" % type(value))
        self._model = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        for feature in value:
            if not isinstance(feature, CRFFeature):
                raise ValueError("Expected CRFFeature but found: %s"
                                 % type(feature))
        self._features = value

    def predict(self, tokens):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")
        features = self.compute_features(tokens)
        predicted_labels = self.model.predict_single(features)
        return remove_bilou_tags(predicted_labels)

    def fit(self, data):
        X = [self.compute_features(sample['tokens']) for sample in data]
        Y = [self.add_bilou(sample['labels']) for sample in data]
        self.model.fit(X, Y)
        self.fitted = True

    def add_bilou(self, labels):
        return add_bilou_tags(labels) if self.use_bilou else labels

    def remove_bilou(self, labels):
        return remove_bilou_tags(labels) if self.use_bilou else labels

    def compute_features(self, tokens):
        features = []
        for token in tokens:
            token_features = dict()
            for feature in self.features:
                token_features.update({feature.name: feature.compute(token)})
            features.append(token_features)
        return features

    def save(self, filepath):
        cPickle.dump(self, open(filepath, 'wb'))

    @classmethod
    def load(cls, filepath):
        return cPickle.load(filepath)
