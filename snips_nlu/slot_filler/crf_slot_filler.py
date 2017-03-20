import cPickle

from sklearn_crfsuite import CRF

from snips_nlu.slot_filler.crf_utils import add_bilou_tags, remove_bilou_tags


class CRFSlotFiller(object):
    def __init__(self, crf_model, tokenization_fn, feature_fns,
                 use_bilou=True):
        self.tokenization_fn = tokenization_fn
        self._crf_model = None
        self.crf_model = crf_model
        self._feature_fns = None
        self.feature_fns = feature_fns
        self.use_bilou = use_bilou
        self.fitted = False

    @property
    def crf_model(self):
        return self._crf_model

    @crf_model.setter
    def crf_model(self, value):
        if not isinstance(value, CRF):
            raise ValueError("Expected CRF but found: %s" % type(value))
        self._crf_model = value

    @property
    def feature_fns(self):
        return self._feature_fns

    @feature_fns.setter
    def feature_fns(self, value):
        for feature_fn in value:
            if not callable(feature_fn):
                raise ValueError("Expected a callable, got %s"
                                 % type(feature_fn))
        self._feature_fns = value

    def get_slots(self, text):
        pass

    def predict(self, tokens):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")
        features = self.compute_features(tokens)
        predicted_labels = self.crf_model.predict_single(features)
        return remove_bilou_tags(predicted_labels)

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
        features = []
        for token in tokens:
            token_features = dict()
            for feature in self.feature_fns:
                token_features.update({feature.name: feature.compute(token)})
            features.append(token_features)
        return features

    def save(self, filepath):
        # TODO: dump as json + use PyCRFSuite serialization
        cPickle.dump(self, open(filepath, 'wb'))

    @classmethod
    def load(cls, filepath):
        return cPickle.load(filepath)
