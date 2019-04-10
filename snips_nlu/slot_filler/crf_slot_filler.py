from __future__ import unicode_literals

import base64
import json
import logging
import math
import shutil
import tempfile
from builtins import range
from copy import deepcopy
from pathlib import Path

from future.utils import iteritems
from sklearn_crfsuite import CRF

from snips_nlu.common.dataset_utils import get_slot_name_mapping
from snips_nlu.common.dict_utils import UnupdatableDict
from snips_nlu.common.io_utils import mkdir_p
from snips_nlu.common.log_utils import DifferedLoggingMessage, log_elapsed_time
from snips_nlu.common.utils import (
    check_persisted_path,
    check_random_state, fitted_required, json_string)
from snips_nlu.constants import (
    DATA, LANGUAGE)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.exceptions import LoadingError
from snips_nlu.pipeline.configs import CRFSlotFillerConfig
from snips_nlu.preprocessing import tokenize
from snips_nlu.slot_filler.crf_utils import (
    OUTSIDE, TAGS, TOKENS, tags_to_slots, utterance_to_sample)
from snips_nlu.slot_filler.feature import TOKEN_NAME
from snips_nlu.slot_filler.feature_factory import CRFFeatureFactory
from snips_nlu.slot_filler.slot_filler import SlotFiller

logger = logging.getLogger(__name__)


@SlotFiller.register("crf_slot_filler")
class CRFSlotFiller(SlotFiller):
    """Slot filler which uses Linear-Chain Conditional Random Fields underneath

    Check https://en.wikipedia.org/wiki/Conditional_random_field to learn
    more about CRFs
    """

    config_type = CRFSlotFillerConfig

    def __init__(self, config=None, **shared):
        """The CRF slot filler can be configured by passing a
        :class:`.CRFSlotFillerConfig`"""
        # The CRFSlotFillerConfig must be deep-copied as it is mutated when
        # fitting the feature factories
        config = deepcopy(config)
        super(CRFSlotFiller, self).__init__(config, **shared)
        self.crf_model = None
        self.features_factories = [
            CRFFeatureFactory.from_config(conf, **shared)
            for conf in self.config.feature_factory_configs]
        self._features = None
        self.language = None
        self.intent = None
        self.slot_name_mapping = None

    @property
    def features(self):
        """List of :class:`.Feature` used by the CRF"""
        if self._features is None:
            self._features = []
            feature_names = set()
            for factory in self.features_factories:
                for feature in factory.build_features():
                    if feature.name in feature_names:
                        raise KeyError("Duplicated feature: %s" % feature.name)
                    feature_names.add(feature.name)
                    self._features.append(feature)
        return self._features

    @property
    def labels(self):
        """List of CRF labels

        These labels differ from the slot names as they contain an additional
        prefix which depends on the :class:`.TaggingScheme` that is used
        (BIO by default).
        """
        labels = []
        if self.crf_model.tagger_ is not None:
            labels = [_decode_tag(label) for label in
                      self.crf_model.tagger_.labels()]
        return labels

    @property
    def fitted(self):
        """Whether or not the slot filler has already been fitted"""
        return self.slot_name_mapping is not None

    @log_elapsed_time(logger, logging.DEBUG,
                      "Fitted CRFSlotFiller in {elapsed_time}")
    # pylint:disable=arguments-differ
    def fit(self, dataset, intent):
        """Fits the slot filler

        Args:
            dataset (dict): A valid Snips dataset
            intent (str): The specific intent of the dataset to train
                the slot filler on

        Returns:
            :class:`CRFSlotFiller`: The same instance, trained
        """
        logger.debug("Fitting %s slot filler...", intent)
        dataset = validate_and_format_dataset(dataset)
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        for factory in self.features_factories:
            factory.custom_entity_parser = self.custom_entity_parser
            factory.builtin_entity_parser = self.builtin_entity_parser
            factory.resources = self.resources

        self.language = dataset[LANGUAGE]
        self.intent = intent
        self.slot_name_mapping = get_slot_name_mapping(dataset, intent)

        if not self.slot_name_mapping:
            # No need to train the CRF if the intent has no slots
            return self

        random_state = check_random_state(self.config.random_seed)
        augmented_intent_utterances = augment_utterances(
            dataset, self.intent, language=self.language,
            resources=self.resources, random_state=random_state,
            **self.config.data_augmentation_config.to_dict())

        crf_samples = [
            utterance_to_sample(u[DATA], self.config.tagging_scheme,
                                self.language)
            for u in augmented_intent_utterances]

        for factory in self.features_factories:
            factory.fit(dataset, intent)

        # Ensure that X, Y are safe and that the OUTSIDE label is learnt to
        # avoid segfault at inference time
        # pylint: disable=C0103
        X = [self.compute_features(sample[TOKENS], drop_out=True)
             for sample in crf_samples]
        Y = [[tag for tag in sample[TAGS]] for sample in crf_samples]
        X, Y = _ensure_safe(X, Y)

        # ensure ascii tags
        Y = [[_encode_tag(tag) for tag in y] for y in Y]

        # pylint: enable=C0103
        self.crf_model = _get_crf_model(self.config.crf_args)
        self.crf_model.fit(X, Y)

        logger.debug(
            "Most relevant features for %s:\n%s", self.intent,
            DifferedLoggingMessage(self.log_weights))
        return self

    # pylint:enable=arguments-differ

    @fitted_required
    def get_slots(self, text):
        """Extracts slots from the provided text

        Returns:
            list of dict: The list of extracted slots

        Raises:
            NotTrained: When the slot filler is not fitted
        """
        if not self.slot_name_mapping:
            # Early return if the intent has no slots
            return []

        tokens = tokenize(text, self.language)
        if not tokens:
            return []
        features = self.compute_features(tokens)
        tags = self.crf_model.predict_single(features)
        logger.debug(DifferedLoggingMessage(
            self.log_inference_weights, text, tokens=tokens, features=features,
            tags=tags))
        decoded_tags = [_decode_tag(t) for t in tags]
        return tags_to_slots(text, tokens, decoded_tags,
                             self.config.tagging_scheme,
                             self.slot_name_mapping)

    def compute_features(self, tokens, drop_out=False):
        """Computes features on the provided tokens

        The *drop_out* parameters allows to activate drop out on features that
        have a positive drop out ratio. This should only be used during
        training.
        """

        cache = [{TOKEN_NAME: token} for token in tokens]
        features = []
        random_state = check_random_state(self.config.random_seed)
        for i in range(len(tokens)):
            token_features = UnupdatableDict()
            for feature in self.features:
                f_drop_out = feature.drop_out
                if drop_out and random_state.rand() < f_drop_out:
                    continue
                value = feature.compute(i, cache)
                if value is not None:
                    token_features[feature.name] = value
            features.append(token_features)
        return features

    @fitted_required
    def get_sequence_probability(self, tokens, labels):
        """Gives the joint probability of a sequence of tokens and CRF labels

        Args:
            tokens (list of :class:`.Token`): list of tokens
            labels (list of str): CRF labels with their tagging scheme prefix
                ("B-color", "I-color", "O", etc)

        Note:
            The absolute value returned here is generally not very useful,
            however it can be used to compare a sequence of labels relatively
            to another one.
        """
        if not self.slot_name_mapping:
            return 0.0 if any(label != OUTSIDE for label in labels) else 1.0
        features = self.compute_features(tokens)
        return self._get_sequence_probability(features, labels)

    @fitted_required
    def _get_sequence_probability(self, features, labels):
        # Use a default substitution label when a label was not seen during
        # training
        substitution_label = OUTSIDE if OUTSIDE in self.labels else \
            self.labels[0]
        cleaned_labels = [
            _encode_tag(substitution_label if l not in self.labels else l)
            for l in labels]
        self.crf_model.tagger_.set(features)
        return self.crf_model.tagger_.probability(cleaned_labels)

    @fitted_required
    def log_weights(self):
        """Returns a logs for both the label-to-label and label-to-features
         weights"""
        if not self.slot_name_mapping:
            return "No weights to display: intent '%s' has no slots" \
                   % self.intent
        log = ""
        transition_features = self.crf_model.transition_features_
        transition_features = sorted(
            iteritems(transition_features), key=_weight_absolute_value,
            reverse=True)
        log += "\nTransition weights: \n\n"
        for (state_1, state_2), weight in transition_features:
            log += "\n%s %s: %s" % (
                _decode_tag(state_1), _decode_tag(state_2), weight)
        feature_weights = self.crf_model.state_features_
        feature_weights = sorted(
            iteritems(feature_weights), key=_weight_absolute_value,
            reverse=True)
        log += "\n\nFeature weights: \n\n"
        for (feat, tag), weight in feature_weights:
            log += "\n%s %s: %s" % (feat, _decode_tag(tag), weight)
        return log

    def log_inference_weights(self, text, tokens, features, tags):
        model_features = set(
            f for (f, _), w in iteritems(self.crf_model.state_features_))
        log = "Feature weights for \"%s\":\n\n" % text
        max_index = len(tokens) - 1
        tokens_logs = []
        for i, (token, feats, tag) in enumerate(zip(tokens, features, tags)):
            token_log = "# Token \"%s\" (tagged as %s):" \
                        % (token.value, _decode_tag(tag))
            if i != 0:
                weights = sorted(self._get_outgoing_weights(tags[i - 1]),
                                 key=_weight_absolute_value, reverse=True)
                if weights:
                    token_log += "\n\nTransition weights from previous tag:"
                    weight_lines = (
                        "- (%s, %s) -> %s"
                        % (_decode_tag(a), _decode_tag(b), w)
                        for (a, b), w in weights
                    )
                    token_log += "\n" + "\n".join(weight_lines)
                else:
                    token_log += \
                        "\n\nNo transition from previous tag seen at" \
                        " train time !"

            if i != max_index:
                weights = sorted(self._get_incoming_weights(tags[i + 1]),
                                 key=_weight_absolute_value, reverse=True)
                if weights:
                    token_log += "\n\nTransition weights to next tag:"
                    weight_lines = (
                        "- (%s, %s) -> %s"
                        % (_decode_tag(a), _decode_tag(b), w)
                        for (a, b), w in weights
                    )
                    token_log += "\n" + "\n".join(weight_lines)
                else:
                    token_log += \
                        "\n\nNo transition to next tag seen at train time !"
            feats = [":".join(f) for f in iteritems(feats)]
            weights = (w for f in feats for w in self._get_feature_weight(f))
            weights = sorted(weights, key=_weight_absolute_value, reverse=True)
            if weights:
                token_log += "\n\nFeature weights:\n"
                token_log += "\n".join(
                    "- (%s, %s) -> %s"
                    % (f, _decode_tag(t), w) for (f, t), w in weights
                )
            else:
                token_log += "\n\nNo feature weights !"

            unseen_features = sorted(
                set(f for f in feats if f not in model_features))
            if unseen_features:
                token_log += "\n\nFeatures not seen at train time:\n%s" % \
                             "\n".join("- %s" % f for f in unseen_features)
            tokens_logs.append(token_log)

        log += "\n\n\n".join(tokens_logs)
        return log

    @fitted_required
    def _get_incoming_weights(self, tag):
        return [((first, second), w) for (first, second), w
                in iteritems(self.crf_model.transition_features_)
                if second == tag]

    @fitted_required
    def _get_outgoing_weights(self, tag):
        return [((first, second), w) for (first, second), w
                in iteritems(self.crf_model.transition_features_)
                if first == tag]

    @fitted_required
    def _get_feature_weight(self, feature):
        return [((f, tag), w) for (f, tag), w
                in iteritems(self.crf_model.state_features_) if f == feature]

    @check_persisted_path
    def persist(self, path):
        """Persists the object at the given path"""
        path.mkdir()

        crf_model_file = None
        if self.crf_model is not None:
            destination = path / Path(self.crf_model.modelfile.name).name
            shutil.copy(self.crf_model.modelfile.name, str(destination))
            crf_model_file = str(destination.name)

        model = {
            "language_code": self.language,
            "intent": self.intent,
            "crf_model_file": crf_model_file,
            "slot_name_mapping": self.slot_name_mapping,
            "config": self.config.to_dict(),
        }
        model_json = json_string(model)
        model_path = path / "slot_filler.json"
        with model_path.open(mode="w") as f:
            f.write(model_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`CRFSlotFiller` instance from a path

        The data at the given path must have been generated using
        :func:`~CRFSlotFiller.persist`
        """
        path = Path(path)
        model_path = path / "slot_filler.json"
        if not model_path.exists():
            raise LoadingError(
                "Missing slot filler model file: %s" % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model = json.load(f)

        slot_filler_config = cls.config_type.from_dict(model["config"])
        slot_filler = cls(config=slot_filler_config, **shared)
        slot_filler.language = model["language_code"]
        slot_filler.intent = model["intent"]
        slot_filler.slot_name_mapping = model["slot_name_mapping"]
        crf_model_file = model["crf_model_file"]
        if crf_model_file is not None:
            crf = _crf_model_from_path(path / crf_model_file)
            slot_filler.crf_model = crf
        return slot_filler

    def __del__(self):
        if self.crf_model is None or self.crf_model.modelfile.name is None:
            return
        try:
            Path(self.crf_model.modelfile.name).unlink()
        except OSError:
            pass


def _get_crf_model(crf_args):
    model_filename = crf_args.get("model_filename", None)
    if model_filename is not None:
        directory = Path(model_filename).parent
        if not directory.is_dir():
            mkdir_p(directory)

    return CRF(model_filename=model_filename, **crf_args)


def _encode_tag(tag):
    return base64.b64encode(tag.encode("utf8"))


def _decode_tag(tag):
    return base64.b64decode(tag).decode("utf8")


def _crf_model_from_path(crf_model_path):
    with crf_model_path.open(mode="rb") as f:
        crf_model_data = f.read()
    with tempfile.NamedTemporaryFile(suffix=".crfsuite", prefix="model",
                                     delete=False) as f:
        f.write(crf_model_data)
        f.flush()
        crf = CRF(model_filename=f.name)
    return crf


# pylint: disable=invalid-name
def _ensure_safe(X, Y):
    """Ensures that Y has at least one not empty label, otherwise the CRF model
    does not contain any label and crashes at

    Args:
        X: features
        Y: labels

    Returns:
        (safe_X, safe_Y): a pair of safe features and labels
    """
    safe_X = list(X)
    safe_Y = list(Y)
    if not any(X) or not any(Y):
        safe_X.append([""])  # empty feature
        safe_Y.append([OUTSIDE])  # outside label
    return safe_X, safe_Y


def _weight_absolute_value(x):
    return math.fabs(x[1])
