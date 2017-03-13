import cPickle

from sklearn_crfsuite import CRF

from snips_nlu.crf.crf_feature import CRFFeature
from snips_nlu.crf.crf_utils import get_bilou_labels
from snips_nlu.tokenizer import Tokenizer


class CRFTokenClassifier:
    def __init__(self, model, tokenizer, features):
        self._model = None
        self.model = model

        self._tokenizer = None
        self.tokenizer = tokenizer

        self._features = None
        self.features = features

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        assert isinstance(value, Tokenizer)
        self._tokenizer = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        assert isinstance(value, CRF)
        self._model = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        assert all(isinstance(feature, CRFFeature) for feature in value)
        self._features = value

    def parse(self, tokens):
        features = self.compute_features(tokens)
        return self.model.predict_single(features)

    def fit(self, utterances):
        tokens_and_labels = [self.get_tokens_and_labels(utterance) for utterance
                             in utterances]
        X = [self.compute_features(sample['tokens']) for sample in
             tokens_and_labels]
        Y = [get_bilou_labels(sample['labels']) for sample in tokens_and_labels]
        self.model.fit(X, Y)

    def get_tokens_and_labels(self, utterance):
        tokens = []
        labels = []
        for chunk in utterance:
            _tokens = self.tokenizer.tokenize(chunk['text'])
            _label = chunk.get('slotName', None)
            _labels = [_label for t in _tokens]
            tokens += _tokens
            labels += _labels

        return {'tokens': tokens, 'labels': labels}

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
