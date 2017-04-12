import importlib
import math
from copy import deepcopy

from sklearn_crfsuite import CRF

from snips_nlu.languages import Language
from snips_nlu.preprocessing import stem
from snips_nlu.slot_filler.crf_utils import TaggingScheme, TOKENS, TAGS
from snips_nlu.slot_filler.feature_functions import (
    TOKEN_NAME, create_feature_function)
from snips_nlu.tokenization import Token
from snips_nlu.utils import (UnupdatableDict, instance_to_generic_dict,
                             ensure_string, safe_pickle_dumps,
                             safe_pickle_loads)

POSSIBLE_SET_FEATURES = ["collection", "common_words"]


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


def scaled_regularization(n_samples, n_reference=50):
    c1, c2 = .0, .0

    coef = n_samples / float(n_reference)
    c1 *= coef
    c2 *= coef

    return c1, c2


class CRFTagger(object):
    def __init__(self, crf_model, features_signatures, tagging_scheme,
                 language):
        self.crf_model = crf_model
        self.features_signatures = features_signatures
        self._features = None
        self.tagging_scheme = tagging_scheme
        self.fitted = False
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

    def fit(self, data, verbose=False):
        X = [self.compute_features(sample[TOKENS]) for sample in data]
        Y = [sample[TAGS] for sample in data]

        c1, c2 = scaled_regularization(len(X))
        self.crf_model.c1 = c1
        self.crf_model.c2 = c2
        self.crf_model = self.crf_model.fit(X, Y)
        self.fitted = True
        if verbose:
            transition_features = self.crf_model.transition_features_
            transition_features = sorted(
                transition_features.iteritems(),
                key=lambda (transition, weight): math.fabs(weight),
                reverse=True)
            print "\nTransition weights: \n\n"
            for (state_1, state_2), weight in transition_features:
                print "%s %s: %s" % (state_1, state_2, weight)

            feature_weights = self.crf_model.state_features_
            feature_weights = sorted(
                feature_weights.iteritems(),
                key=lambda (feature, weight): math.fabs(weight),
                reverse=True)
            print "\nFeature weights: \n\n"
            for (feat, tag), weight in feature_weights:
                print "%s %s: %s" % (feat, tag, weight)

        return self

    def compute_features(self, tokens):
        tokens = [
            Token(t.value, t.start, t.end, stem=stem(t.value, self.language))
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
        features_signatures = deepcopy(self.features_signatures)

        for signature in features_signatures:
            for feat in POSSIBLE_SET_FEATURES:
                if feat in signature["args"] and isinstance(
                        signature["args"][feat], set):
                    signature["args"][feat] = list(signature["args"][feat])

        obj_dict.update({
            "crf_model": safe_pickle_dumps(self.crf_model),
            "features_signatures": features_signatures,
            "tagging_scheme": self.tagging_scheme.value,
            "fitted": self.fitted,
            "language": self.language.iso_code
        })
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        crf_model = safe_pickle_loads(ensure_string(obj_dict["crf_model"]))
        features_signatures = deepcopy(obj_dict["features_signatures"])
        for signature in features_signatures:
            for feat in POSSIBLE_SET_FEATURES:
                if feat in signature["args"] and isinstance(
                        signature["args"][feat], list):
                    signature["args"][feat] = set(signature["args"][feat])

        tagging_scheme = TaggingScheme(int(obj_dict["tagging_scheme"]))
        language = Language.from_iso_code(obj_dict["language"])
        fitted = obj_dict["fitted"]
        self = cls(crf_model=crf_model,
                   features_signatures=features_signatures,
                   tagging_scheme=tagging_scheme, language=language)
        self.fitted = fitted
        return self

    def __eq__(self, other):
        if not isinstance(other, CRFTagger):
            return False
        self_model_state = self.crf_model.__getstate__()
        other_model_state = other.crf_model.__getstate__()
        self_model_state.pop('modelfile')
        other_model_state.pop('modelfile')
        return self.features_signatures == other.features_signatures \
               and self.tagging_scheme == other.tagging_scheme \
               and self.fitted == other.fitted \
               and self.language == other.language \
               and self_model_state == other_model_state

    def __ne__(self, other):
        return not self.__eq__(other)
