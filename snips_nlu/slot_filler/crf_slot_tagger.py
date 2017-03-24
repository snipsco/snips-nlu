from sklearn_crfsuite import CRF

from feature_functions import TOKEN_NAME
from snips_nlu.utils import UnupdatableDict


def default_crf_model():
    return CRF(min_freq=None, c1=.1, c2=.1, max_iterations=None, verbose=False)


class CRFTagger(object):
    def __init__(self, crf_model, features, tagging):
        self.crf_model = crf_model
        self.features = features
        self.tagging = tagging
        self.fitted = False

    def get_tags(self, tokens):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")
        features = self.compute_features(tokens)
        return self.crf_model.predict_single(features)

    def fit(self, data):
        X = [self.compute_features(sample['tokens']) for sample in data]
        Y = [sample['tags'] for sample in data]
        self.crf_model.fit(X, Y)
        self.fitted = True
        return self

    def compute_features(self, tokens):
        cache = [{TOKEN_NAME: token.value} for token in tokens]
        features = []
        for i in range(len(tokens)):
            token_features = UnupdatableDict()
            for feature_name, feature_fn in self.features:
                value = feature_fn(i, cache)
                if value is not None:
                    token_features[feature_name] = value
            features.append(token_features)
        return features
