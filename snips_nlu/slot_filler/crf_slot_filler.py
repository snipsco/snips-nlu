from sklearn_crfsuite import CRF

from feature_functions import TOKEN_NAME
from snips_nlu.slot_filler.crf_utils import (add_bilou_tags, remove_bilou_tags,
                                             OUTSIDE)
from snips_nlu.utils import UnupdatableDict


def default_crf_model():
    return CRF(min_freq=None, c1=.1, c2=.1, max_iterations=None, verbose=False)


class CRFSlotFiller(object):
    def __init__(self, crf_model, features, use_bilou=True):
        self.crf_model = crf_model
        self.features = features
        self.use_bilou = use_bilou
        self.fitted = False

    def get_slots(self, tokens):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")
        features = self.compute_features(tokens)
        predicted_labels = self.crf_model.predict_single(features)
        predicted_labels = self.remove_bilou(predicted_labels)
        predicted_labels = map(lambda l: l if l != OUTSIDE else None,
                               predicted_labels)
        return predicted_labels

    def fit(self, data):
        X = [self.compute_features(sample['tokens']) for sample in data]
        Y = [self.add_bilou(sample['labels']) for sample in data]
        self.crf_model.fit(X, Y)
        self.fitted = True

    def add_bilou(self, labels):
        return add_bilou_tags(labels) if self.use_bilou else labels

    def remove_bilou(self, labels):
        return remove_bilou_tags(labels) if self.use_bilou else labels

    def compute_features(self, tokens):
        cache = [{TOKEN_NAME: token} for token in tokens]
        features = []
        for i in range(len(tokens)):
            token_features = UnupdatableDict()
            for feature_name, feature_fn in self.features:
                value = feature_fn(i, cache)
                if value is not None:
                    token_features[feature_name] = value
            features.append(token_features)
        return features
