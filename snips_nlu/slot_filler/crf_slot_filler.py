from __future__ import unicode_literals

import base64
import json
import logging
import math
import shutil
import tempfile
from builtins import range
from copy import copy
from itertools import groupby, product
from pathlib import Path

from future.utils import iteritems
from sklearn_crfsuite import CRF

from snips_nlu.constants import (
    DATA, END, ENTITY_KIND, LANGUAGE, RES_ENTITY,
    RES_MATCH_RANGE, RES_VALUE, START)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.pipeline.configs import CRFSlotFillerConfig
from snips_nlu.preprocessing import tokenize
from snips_nlu.slot_filler.crf_utils import (
    OUTSIDE, TAGS, TOKENS, positive_tagging, tag_name_to_slot_name,
    tags_to_preslots, tags_to_slots, utterance_to_sample)
from snips_nlu.slot_filler.feature import TOKEN_NAME
from snips_nlu.slot_filler.feature_factory import get_feature_factory
from snips_nlu.slot_filler.slot_filler import SlotFiller
from snips_nlu.utils import (
    DifferedLoggingMessage, UnupdatableDict, check_persisted_path,
    check_random_state, fitted_required, get_slot_name_mapping, json_string,
    log_elapsed_time,
    mkdir_p, ranges_overlap)

logger = logging.getLogger(__name__)


class CRFSlotFiller(SlotFiller):
    """Slot filler which uses Linear-Chain Conditional Random Fields underneath

    Check https://en.wikipedia.org/wiki/Conditional_random_field to learn
    more about CRFs
    """

    unit_name = "crf_slot_filler"
    config_type = CRFSlotFillerConfig

    def __init__(self, config=None, **shared):
        """The CRF slot filler can be configured by passing a
        :class:`.CRFSlotFillerConfig`"""

        if config is None:
            config = self.config_type()
        super(CRFSlotFiller, self).__init__(config, **shared)
        self.crf_model = None
        self.features_factories = [get_feature_factory(conf) for conf in
                                   self.config.feature_factory_configs]
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
                for feature in factory.build_features(
                        self.builtin_entity_parser, self.custom_entity_parser):
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
        """Fit the slot filler

        Args:
            dataset (dict): A valid Snips dataset
            intent (str): The specific intent of the dataset to train
                the slot filler on

        Returns:
            :class:`CRFSlotFiller`: The same instance, trained
        """
        logger.debug("Fitting %s slot filler...", intent)
        dataset = validate_and_format_dataset(dataset)
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        self.language = dataset[LANGUAGE]
        self.intent = intent
        self.slot_name_mapping = get_slot_name_mapping(dataset, intent)

        if not self.slot_name_mapping:
            # No need to train the CRF if the intent has no slots
            return self

        random_state = check_random_state(self.config.random_seed)
        augmented_intent_utterances = augment_utterances(
            dataset, self.intent, language=self.language,
            random_state=random_state,
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
        tags = [_decode_tag(tag) for tag in
                self.crf_model.predict_single(features)]
        slots = tags_to_slots(text, tokens, tags, self.config.tagging_scheme,
                              self.slot_name_mapping)

        builtin_slots_names = set(slot_name for (slot_name, entity) in
                                  iteritems(self.slot_name_mapping)
                                  if is_builtin_entity(entity))
        if not builtin_slots_names:
            return slots

        # Replace tags corresponding to builtin entities by outside tags
        tags = _replace_builtin_tags(tags, builtin_slots_names)
        return self._augment_slots(text, tokens, tags, builtin_slots_names)

    def compute_features(self, tokens, drop_out=False):
        """Compute features on the provided tokens

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
        """Return a logs for both the label-to-label and label-to-features
         weights"""
        if not self.slot_name_mapping:
            return "No weights to display: intent '%s' has no slots" \
                   % self.intent
        log = ""
        transition_features = self.crf_model.transition_features_
        transition_features = sorted(
            iteritems(transition_features),
            key=lambda transition_weight: math.fabs(transition_weight[1]),
            reverse=True)
        log += "\nTransition weights: \n\n"
        for (state_1, state_2), weight in transition_features:
            log += "\n%s %s: %s" % (
                _decode_tag(state_1), _decode_tag(state_2), weight)
        feature_weights = self.crf_model.state_features_
        feature_weights = sorted(
            iteritems(feature_weights),
            key=lambda feature_weight: math.fabs(feature_weight[1]),
            reverse=True)
        log += "\n\nFeature weights: \n\n"
        for (feat, tag), weight in feature_weights:
            log += "\n%s %s: %s" % (feat, _decode_tag(tag), weight)
        return log

    def _augment_slots(self, text, tokens, tags, builtin_slots_names):
        scope = set(self.slot_name_mapping[slot]
                    for slot in builtin_slots_names)
        builtin_entities = [
            be for entity_kind in scope for be in
            self.builtin_entity_parser.parse(text, scope=[entity_kind],
                                             use_cache=True)]
        # We remove builtin entities which conflicts with custom slots
        # extracted by the CRF
        builtin_entities = _filter_overlapping_builtins(
            builtin_entities, tokens, tags, self.config.tagging_scheme)

        # We resolve conflicts between builtin entities by keeping the longest
        # matches. In case when two builtin entities span the same range, we
        # keep both.
        builtin_entities = _disambiguate_builtin_entities(builtin_entities)

        # We group builtin entities based on their position
        grouped_entities = (
            list(bes)
            for _, bes in groupby(builtin_entities,
                                  key=lambda s: s[RES_MATCH_RANGE][START]))
        grouped_entities = sorted(
            grouped_entities,
            key=lambda entities: entities[0][RES_MATCH_RANGE][START])

        features = self.compute_features(tokens)
        spans_ranges = [entities[0][RES_MATCH_RANGE]
                        for entities in grouped_entities]
        tokens_indexes = _spans_to_tokens_indexes(spans_ranges, tokens)

        # We loop on all possible slots permutations and use the CRF to find
        # the best one in terms of probability
        slots_permutations = _get_slots_permutations(
            grouped_entities, self.slot_name_mapping)
        best_updated_tags = tags
        best_permutation_score = -1
        for slots in slots_permutations:
            updated_tags = copy(tags)
            for slot_index, slot in enumerate(slots):
                indexes = tokens_indexes[slot_index]
                sub_tags_sequence = positive_tagging(
                    self.config.tagging_scheme, slot, len(indexes))
                updated_tags[indexes[0]:indexes[-1] + 1] = sub_tags_sequence
            score = self._get_sequence_probability(features, updated_tags)
            if score > best_permutation_score:
                best_updated_tags = updated_tags
                best_permutation_score = score
        slots = tags_to_slots(text, tokens, best_updated_tags,
                              self.config.tagging_scheme,
                              self.slot_name_mapping)

        return _reconciliate_builtin_slots(text, slots, builtin_entities)

    @check_persisted_path
    def persist(self, path):
        """Persist the object at the given path"""
        path = Path(path)
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
        """Load a :class:`CRFSlotFiller` instance from a path

        The data at the given path must have been generated using
        :func:`~CRFSlotFiller.persist`
        """
        path = Path(path)
        model_path = path / "slot_filler.json"
        if not model_path.exists():
            raise OSError("Missing slot filler model file: %s"
                          % model_path.name)

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


def _replace_builtin_tags(tags, builtin_slot_names):
    new_tags = []
    for tag in tags:
        if tag == OUTSIDE:
            new_tags.append(tag)
        else:
            slot_name = tag_name_to_slot_name(tag)
            if slot_name in builtin_slot_names:
                new_tags.append(OUTSIDE)
            else:
                new_tags.append(tag)
    return new_tags


def _filter_overlapping_builtins(builtin_entities, tokens, tags,
                                 tagging_scheme):
    slots = tags_to_preslots(tokens, tags, tagging_scheme)
    ents = []
    for ent in builtin_entities:
        if any(ranges_overlap(ent[RES_MATCH_RANGE], s[RES_MATCH_RANGE])
               for s in slots):
            continue
        ents.append(ent)
    return ents


def _spans_to_tokens_indexes(spans, tokens):
    tokens_indexes = []
    for span in spans:
        indexes = []
        for i, token in enumerate(tokens):
            if span[END] > token.start and span[START] < token.end:
                indexes.append(i)
        tokens_indexes.append(indexes)
    return tokens_indexes


def _reconciliate_builtin_slots(text, slots, builtin_entities):
    for slot in slots:
        if not is_builtin_entity(slot[RES_ENTITY]):
            continue
        for be in builtin_entities:
            if be[ENTITY_KIND] != slot[RES_ENTITY]:
                continue
            be_start = be[RES_MATCH_RANGE][START]
            be_end = be[RES_MATCH_RANGE][END]
            be_length = be_end - be_start
            slot_start = slot[RES_MATCH_RANGE][START]
            slot_end = slot[RES_MATCH_RANGE][END]
            slot_length = slot_end - slot_start
            if be_start <= slot_start and be_end >= slot_end \
                    and be_length > slot_length:
                slot[RES_MATCH_RANGE] = {
                    START: be_start,
                    END: be_end
                }
                slot[RES_VALUE] = text[be_start: be_end]
                break
    return slots


def _disambiguate_builtin_entities(builtin_entities):
    if not builtin_entities:
        return []
    builtin_entities = sorted(
        builtin_entities,
        key=lambda be: be[RES_MATCH_RANGE][END] - be[RES_MATCH_RANGE][START],
        reverse=True)

    disambiguated_entities = [builtin_entities[0]]
    for entity in builtin_entities[1:]:
        entity_rng = entity[RES_MATCH_RANGE]
        conflict = False
        for disambiguated_entity in disambiguated_entities:
            disambiguated_entity_rng = disambiguated_entity[RES_MATCH_RANGE]
            if ranges_overlap(entity_rng, disambiguated_entity_rng):
                conflict = True
                if entity_rng == disambiguated_entity_rng:
                    disambiguated_entities.append(entity)
                break
        if not conflict:
            disambiguated_entities.append(entity)

    return sorted(disambiguated_entities,
                  key=lambda be: be[RES_MATCH_RANGE][START])


def _get_slots_permutations(grouped_entities, slot_name_mapping):
    # We associate to each group of entities the list of slot names that
    # could correspond
    possible_slots = [
        list(set(slot_name for slot_name, ent in iteritems(slot_name_mapping)
                 for entity in entities if ent == entity[ENTITY_KIND]))
        + [OUTSIDE]
        for entities in grouped_entities]
    return product(*possible_slots)


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
    """Ensure that Y has at least one not empty label, otherwise the CRF model
    does not contain any label and crashes at
    Args:
        X: features
        Y: labels

    Returns: (safe_X, safe_Y) a pair of safe features and labels

    """
    safe_X = list(X)
    safe_Y = list(Y)
    if not any(X) or not any(Y):
        safe_X.append([""])  # empty feature
        safe_Y.append([OUTSIDE])  # outside label
    return safe_X, safe_Y
