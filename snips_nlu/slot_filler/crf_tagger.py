import cPickle
import importlib

from sklearn_crfsuite import CRF

from snips_nlu.preprocessing import stem
from snips_nlu.slot_filler.crf_utils import Tagging, TOKENS, TAGS
from snips_nlu.slot_filler.feature_functions import TOKEN_NAME, \
    create_feature_function
from snips_nlu.tokenization import Token
from snips_nlu.utils import UnupdatableDict, instance_to_generic_dict


def default_crf_model():
    return CRF(min_freq=None, c1=.1, c2=.1, max_iterations=None, verbose=False)


def get_features_from_signatures(signatures):
    features = dict()
    for signature in signatures:
        factory_name = signature["factory_name"]
        module_name = signature["module_name"]
        factory = getattr(importlib.import_module(module_name), factory_name)
        fn = factory(**(signature["args"]))
        for offset in signature["offsets"]:
            feature_name, feature_fn = create_feature_function(fn, offset)
            if feature_name in features:
                raise KeyError("Existing feature: %s" % feature_name)
            features[feature_name] = feature_fn
    return features


class CRFTagger(object):
    def __init__(self, crf_model, features_signatures, tagging,
                 use_stemming=False, language=None):
        self.crf_model = crf_model
        self.features_signatures = features_signatures
        self._features = None
        self.tagging = tagging
        self.fitted = False
        if use_stemming and language is None:
            raise ValueError("language must be provided if stemming is used")
        self.use_stemming = use_stemming
        self.language = language

    @property
    def features(self):
        if self._features is None:
            self._features = get_features_from_signatures(
                self.features_signatures)
        return self._features

    def get_tags(self, tokens):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")
        features = self.compute_features(tokens)
        return self.crf_model.predict_single(features)

    def fit(self, data):
        X = [self.compute_features(sample[TOKENS]) for sample in data]
        Y = [sample[TAGS] for sample in data]
        self.crf_model.fit(X, Y)
        self.fitted = True
        return self

    def compute_features(self, tokens):
        if self.use_stemming:
            tokens = [Token(t.value, t.start, t.end,
                            stem=stem(t.value, self.language, t.value))
                      for t in tokens]
        cache = [{TOKEN_NAME: token} for token in tokens]
        features = []
        for i in range(len(tokens)):
            token_features = UnupdatableDict()
            for feature_name, feature_fn in self.features.iteritems():
                value = feature_fn(i, cache)
                if value is not None:
                    token_features[feature_name] = value
            features.append(token_features)
        return features

    def to_dict(self):
        obj_dict = instance_to_generic_dict(self)
        obj_dict.update({
            "crf_model": cPickle.dumps(self.crf_model),
            "features_signatures": self.features_signatures,
            "tagging": self.tagging.value,
            "fitted": self.fitted
        })
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        crf_model = cPickle.loads(obj_dict["crf_model"])
        features_signatures = obj_dict["features_signatures"]
        tagging = Tagging(int(obj_dict["tagging"]))
        fitted = obj_dict["fitted"]
        self = cls(crf_model=crf_model,
                   features_signatures=features_signatures, tagging=tagging)
        self.fitted = fitted
        return self
