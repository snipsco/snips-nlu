from __future__ import unicode_literals

import base64
import io
import math
import os
import tempfile
from copy import deepcopy

from sklearn_crfsuite import CRF

import snips_nlu.slot_filler.feature_functions
from snips_nlu.languages import Language
from snips_nlu.preprocessing import stem
from snips_nlu.slot_filler.crf_utils import TaggingScheme, TOKENS, TAGS, \
    OUTSIDE
from snips_nlu.slot_filler.feature_functions import (
    TOKEN_NAME, create_feature_function)
from snips_nlu.tokenization import Token
from snips_nlu.utils import (UnupdatableDict, mkdir_p)

POSSIBLE_SET_FEATURES = ["collection"]


def default_crf_model(model_filename=None):
    if model_filename is not None:
        directory = os.path.dirname(model_filename)
        if not os.path.isdir(directory):
            mkdir_p(directory)

    return CRF(min_freq=None, c1=.1, c2=.1, max_iterations=None, verbose=False,
               model_filename=model_filename)


def get_features_from_signatures(signatures):
    features = dict()
    for signature in signatures:
        factory_name = signature["factory_name"]
        factory = getattr(snips_nlu.slot_filler.feature_functions,
                          factory_name)
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
        self.language = language

    @property
    def features(self):
        if self._features is None:
            self._features = get_features_from_signatures(
                self.features_signatures)
        return self._features

    @property
    def labels(self):
        if self.crf_model.tagger_ is not None:
            return [label.decode('utf8') for label in
                    self.crf_model.tagger_.labels()]
        else:
            return []

    @property
    def fitted(self):
        return self.crf_model.tagger_ is not None

    def get_tags(self, tokens):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")
        features = self.compute_features(tokens)
        return [tag.decode('utf8') for tag in
                self.crf_model.predict_single(features)]

    def get_sequence_probability(self, tokens, labels):
        if not self.fitted:
            raise AssertionError("Model must be fitted before using predict")

        # Use a default substitution label when a label was not seen during
        # training
        substitution_label = OUTSIDE if OUTSIDE in self.labels else \
            self.labels[0]
        cleaned_labels = [substitution_label if l not in self.labels else l for
                          l in labels]
        cleaned_labels = [label.encode('utf8') for label in cleaned_labels]

        features = self.compute_features(tokens)
        self.crf_model.tagger_.set(features)
        return self.crf_model.tagger_.probability(cleaned_labels)

    def fit(self, data, verbose=False):
        X = [self.compute_features(sample[TOKENS]) for sample in data]
        Y = [[tag.encode('utf8') for tag in sample[TAGS]] for sample in data]

        c1, c2 = scaled_regularization(len(X))
        self.crf_model.c1 = c1
        self.crf_model.c2 = c2
        self.crf_model = self.crf_model.fit(X, Y)
        if verbose:
            transition_features = self.crf_model.transition_features_
            transition_features = sorted(
                transition_features.iteritems(),
                key=lambda (transition, _weight): math.fabs(_weight),
                reverse=True)
            print "\nTransition weights: \n\n"
            for (state_1, state_2), weight in transition_features:
                print "%s %s: %s" % (state_1, state_2, weight)

            feature_weights = self.crf_model.state_features_
            feature_weights = sorted(
                feature_weights.iteritems(),
                key=lambda (feature, _weight): math.fabs(_weight),
                reverse=True)
            print "\nFeature weights: \n\n"
            for (feat, tag), weight in feature_weights:
                print "%s %s: %s" % (feat, tag, weight)

        return self

    def compute_features(self, tokens):
        tokens = [
            Token(t.value, t.start, t.end,
                  stem=stem(t.normalized_value, self.language))
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
        features_signatures = deepcopy(self.features_signatures)

        for signature in features_signatures:
            for feat in POSSIBLE_SET_FEATURES:
                if feat in signature["args"] and isinstance(
                        signature["args"][feat], set):
                    signature["args"][feat] = list(signature["args"][feat])

        return {
            "crf_model_data": serialize_crf_model(self.crf_model),
            "features_signatures": features_signatures,
            "tagging_scheme": self.tagging_scheme.value,
            "language_code": self.language.iso_code
        }

    @classmethod
    def from_dict(cls, tagger_config):
        features_signatures = tagger_config["features_signatures"]
        tagging_scheme = TaggingScheme(int(tagger_config["tagging_scheme"]))
        language = Language.from_iso_code(tagger_config["language_code"])
        crf = deserialize_crf_model(tagger_config["crf_model_data"])
        return cls(crf_model=crf, features_signatures=features_signatures,
                   tagging_scheme=tagging_scheme, language=language)

    def __del__(self):
        if self.crf_model is None or self.crf_model.modelfile.auto \
                or self.crf_model.modelfile.name is None:
            return
        try:
            os.remove(self.crf_model.modelfile.name)
        except OSError:
            pass


def serialize_crf_model(crf_model):
    with io.open(crf_model.modelfile.name, mode='rb') as f:
        crfsuite_data = base64.b64encode(f.read()).decode('ascii')
    return crfsuite_data


def deserialize_crf_model(crf_model_data):
    b64_data = base64.b64decode(crf_model_data)
    with tempfile.NamedTemporaryFile(suffix=".crfsuite", prefix="model",
                                     delete=False) as f:
        f.write(b64_data)
        f.flush()
        crf = CRF(model_filename=f.name)
    return crf
