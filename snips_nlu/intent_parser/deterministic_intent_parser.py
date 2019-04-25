from __future__ import unicode_literals

import json
import logging
import re
from builtins import str
from collections import defaultdict
from pathlib import Path

from future.utils import iteritems, iterkeys, itervalues
from snips_nlu_utils import normalize

from snips_nlu.common.dataset_utils import get_slot_name_mappings
from snips_nlu.common.log_utils import log_elapsed_time, log_result
from snips_nlu.common.utils import (
    check_persisted_path, deduplicate_overlapping_items, fitted_required,
    json_string, ranges_overlap, regex_escape,
    replace_entities_with_placeholders)
from snips_nlu.constants import (
    DATA, END, ENTITIES, ENTITY,
    INTENTS, LANGUAGE, RES_INTENT, RES_INTENT_NAME,
    RES_MATCH_RANGE, RES_SLOTS, RES_VALUE, SLOT_NAME, START, TEXT, UTTERANCES,
    RES_PROBA)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.dataset.utils import extract_entity_values
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.exceptions import IntentNotFoundError, LoadingError
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.pipeline.configs import DeterministicIntentParserConfig
from snips_nlu.preprocessing import normalize_token, tokenize, tokenize_light
from snips_nlu.resources import get_stop_words
from snips_nlu.result import (empty_result, extraction_result,
                              intent_classification_result, parsing_result,
                              unresolved_slot)

WHITESPACE_PATTERN = r"\s*"

logger = logging.getLogger(__name__)


@IntentParser.register("deterministic_intent_parser")
class DeterministicIntentParser(IntentParser):
    """Intent parser using pattern matching in a deterministic manner

    This intent parser is very strict by nature, and tends to have a very good
    precision but a low recall. For this reason, it is interesting to use it
    first before potentially falling back to another parser.
    """

    config_type = DeterministicIntentParserConfig

    def __init__(self, config=None, **shared):
        """The deterministic intent parser can be configured by passing a
        :class:`.DeterministicIntentParserConfig`"""
        super(DeterministicIntentParser, self).__init__(config, **shared)
        self._language = None
        self._slot_names_to_entities = None
        self._group_names_to_slot_names = None
        self._stop_words = None
        self._stop_words_whitelist = None
        self.slot_names_to_group_names = None
        self.regexes_per_intent = None
        self.entity_scopes = None

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        if value is None:
            self._stop_words = None
        else:
            if self.config.ignore_stop_words:
                self._stop_words = get_stop_words(self.resources)
            else:
                self._stop_words = set()

    @property
    def slot_names_to_entities(self):
        return self._slot_names_to_entities

    @slot_names_to_entities.setter
    def slot_names_to_entities(self, value):
        self._slot_names_to_entities = value
        if value is None:
            self.entity_scopes = None
        else:
            self.entity_scopes = {
                intent: {
                    "builtin": {ent for ent in itervalues(slot_mapping)
                                if is_builtin_entity(ent)},
                    "custom": {ent for ent in itervalues(slot_mapping)
                               if not is_builtin_entity(ent)}
                }
                for intent, slot_mapping in iteritems(value)}

    @property
    def group_names_to_slot_names(self):
        return self._group_names_to_slot_names

    @group_names_to_slot_names.setter
    def group_names_to_slot_names(self, value):
        self._group_names_to_slot_names = value
        if value is not None:
            self.slot_names_to_group_names = {
                slot_name: group for group, slot_name in iteritems(value)}

    @property
    def patterns(self):
        """Dictionary of patterns per intent"""
        if self.regexes_per_intent is not None:
            return {i: [r.pattern for r in regex_list] for i, regex_list in
                    iteritems(self.regexes_per_intent)}
        return None

    @patterns.setter
    def patterns(self, value):
        if value is not None:
            self.regexes_per_intent = dict()
            for intent, pattern_list in iteritems(value):
                regexes = [re.compile(r"%s" % p, re.IGNORECASE)
                           for p in pattern_list]
                self.regexes_per_intent[intent] = regexes

    @property
    def fitted(self):
        """Whether or not the intent parser has already been trained"""
        return self.regexes_per_intent is not None

    @log_elapsed_time(
        logger, logging.INFO, "Fitted deterministic parser in {elapsed_time}")
    def fit(self, dataset, force_retrain=True):
        """Fits the intent parser with a valid Snips dataset"""
        logger.info("Fitting deterministic parser...")
        dataset = validate_and_format_dataset(dataset)
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        self.language = dataset[LANGUAGE]
        self.regexes_per_intent = dict()
        entity_placeholders = _get_entity_placeholders(dataset, self.language)
        self.slot_names_to_entities = get_slot_name_mappings(dataset)
        self.group_names_to_slot_names = _get_group_names_to_slot_names(
            self.slot_names_to_entities)
        self._stop_words_whitelist = _get_stop_words_whitelist(
            dataset, self._stop_words)

        # Do not use ambiguous patterns that appear in more than one intent
        all_patterns = set()
        ambiguous_patterns = set()
        intent_patterns = dict()
        for intent_name, intent in iteritems(dataset[INTENTS]):
            patterns = self._generate_patterns(intent_name, intent[UTTERANCES],
                                               entity_placeholders)
            patterns = [p for p in patterns
                        if len(p) < self.config.max_pattern_length]
            existing_patterns = {p for p in patterns if p in all_patterns}
            ambiguous_patterns.update(existing_patterns)
            all_patterns.update(set(patterns))
            intent_patterns[intent_name] = patterns

        for intent_name, patterns in iteritems(intent_patterns):
            patterns = [p for p in patterns if p not in ambiguous_patterns]
            patterns = patterns[:self.config.max_queries]
            regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
            self.regexes_per_intent[intent_name] = regexes
        return self

    @log_result(
        logger, logging.DEBUG, "DeterministicIntentParser result -> {result}")
    @log_elapsed_time(logger, logging.DEBUG, "Parsed in {elapsed_time}.")
    @fitted_required
    def parse(self, text, intents=None, top_n=None):
        """Performs intent parsing on the provided *text*

        Intent and slots are extracted simultaneously through pattern matching

        Args:
            text (str): input
            intents (str or list of str): if provided, reduces the scope of
                intent parsing to the provided list of intents
            top_n (int, optional): when provided, this method will return a
                list of at most top_n most likely intents, instead of a single
                parsing result.
                Note that the returned list can contain less than ``top_n``
                elements, for instance when the parameter ``intents`` is not
                None, or when ``top_n`` is greater than the total number of
                intents.

        Returns:
            dict or list: the most likely intent(s) along with the extracted
            slots. See :func:`.parsing_result` and :func:`.extraction_result`
            for the output format.

        Raises:
            NotTrained: when the intent parser is not fitted
        """
        if top_n is None:
            top_intents = self._parse_top_intents(text, top_n=1,
                                                  intents=intents)
            if top_intents:
                intent = top_intents[0][RES_INTENT]
                slots = top_intents[0][RES_SLOTS]
                if intent[RES_PROBA] < 1.0:
                    # return None in case of ambiguity
                    return empty_result(text, probability=1.0)
                return parsing_result(text, intent, slots)
            return empty_result(text, probability=1.0)
        return self._parse_top_intents(text, top_n=top_n, intents=intents)

    def _parse_top_intents(self, text, top_n, intents=None):
        if isinstance(intents, str):
            intents = {intents}
        elif isinstance(intents, list):
            intents = set(intents)

        if top_n < 1:
            raise ValueError(
                "top_n argument must be greater or equal to 1, but got: %s"
                % top_n)

        def placeholder_fn(entity_name):
            return _get_entity_name_placeholder(entity_name, self.language)

        results = []

        for intent, entity_scope in iteritems(self.entity_scopes):
            if intents is not None and intent not in intents:
                continue
            builtin_entities = self.builtin_entity_parser.parse(
                text, scope=entity_scope["builtin"], use_cache=True)
            custom_entities = self.custom_entity_parser.parse(
                text, scope=entity_scope["custom"], use_cache=True)
            all_entities = builtin_entities + custom_entities
            mapping, processed_text = replace_entities_with_placeholders(
                text, all_entities, placeholder_fn=placeholder_fn)
            cleaned_text = self._preprocess_text(text, intent)
            cleaned_processed_text = self._preprocess_text(processed_text,
                                                           intent)
            for regex in self.regexes_per_intent[intent]:
                res = self._get_matching_result(text, cleaned_processed_text,
                                                regex, intent, mapping)
                if res is None and cleaned_text != cleaned_processed_text:
                    res = self._get_matching_result(text, cleaned_text, regex,
                                                    intent)
                if res is not None:
                    results.append(res)
                    break

        confidence_score = 1.
        if results:
            confidence_score = 1. / float(len(results))

        results = results[:top_n]

        for res in results:
            res[RES_INTENT][RES_PROBA] = confidence_score

        return results

    @fitted_required
    def get_intents(self, text):
        """Returns the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent
        """
        nb_intents = len(self.regexes_per_intent)
        top_intents = [intent_result[RES_INTENT] for intent_result in
                       self._parse_top_intents(text, top_n=nb_intents)]
        matched_intents = {res[RES_INTENT_NAME] for res in top_intents}
        for intent in self.regexes_per_intent:
            if intent not in matched_intents:
                top_intents.append(intent_classification_result(intent, 0.0))

        # The None intent is not included in the regex patterns and is thus
        # never matched by the deterministic parser
        top_intents.append(intent_classification_result(None, 0.0))
        return top_intents

    @fitted_required
    def get_slots(self, text, intent):
        """Extracts slots from a text input, with the knowledge of the intent

        Args:
            text (str): input
            intent (str): the intent which the input corresponds to

        Returns:
            list: the list of extracted slots

        Raises:
            IntentNotFoundError: When the intent was not part of the training
                data
        """
        if intent is None:
            return []

        if intent not in self.regexes_per_intent:
            raise IntentNotFoundError(intent)
        slots = self.parse(text, intents=[intent])[RES_SLOTS]
        if slots is None:
            slots = []
        return slots

    def _get_intent_stop_words(self, intent):
        whitelist = self._stop_words_whitelist.get(intent, set())
        return self._stop_words.difference(whitelist)

    def _preprocess_text(self, string, intent):
        """Replaces stop words and characters that are tokenized out by
            whitespaces"""
        tokens = tokenize(string, self.language)
        current_idx = 0
        cleaned_string = ""
        stop_words = self._get_intent_stop_words(intent)
        for token in tokens:
            if stop_words and normalize_token(token) in stop_words:
                token.value = "".join(" " for _ in range(len(token.value)))
            prefix_length = token.start - current_idx
            cleaned_string += "".join((" " for _ in range(prefix_length)))
            cleaned_string += token.value
            current_idx = token.end
        suffix_length = len(string) - current_idx
        cleaned_string += "".join((" " for _ in range(suffix_length)))
        return cleaned_string

    def _get_matching_result(self, text, processed_text, regex, intent,
                             entities_ranges_mapping=None):
        found_result = regex.match(processed_text)
        if found_result is None:
            return None
        parsed_intent = intent_classification_result(intent_name=intent,
                                                     probability=1.0)
        slots = []
        for group_name in found_result.groupdict():
            ref_group_name = group_name
            if "_" in group_name:
                ref_group_name = group_name.split("_")[0]
            slot_name = self.group_names_to_slot_names[ref_group_name]
            entity = self.slot_names_to_entities[intent][slot_name]
            rng = (found_result.start(group_name),
                   found_result.end(group_name))
            if entities_ranges_mapping is not None:
                if rng in entities_ranges_mapping:
                    rng = entities_ranges_mapping[rng]
                else:
                    shift = _get_range_shift(
                        rng, entities_ranges_mapping)
                    rng = {START: rng[0] + shift, END: rng[1] + shift}
            else:
                rng = {START: rng[0], END: rng[1]}
            value = text[rng[START]:rng[END]]
            parsed_slot = unresolved_slot(
                match_range=rng, value=value, entity=entity,
                slot_name=slot_name)
            slots.append(parsed_slot)
        parsed_slots = _deduplicate_overlapping_slots(slots, self.language)
        parsed_slots = sorted(parsed_slots,
                              key=lambda s: s[RES_MATCH_RANGE][START])
        return extraction_result(parsed_intent, parsed_slots)

    def _generate_patterns(self, intent, intent_utterances,
                           entity_placeholders):
        unique_patterns = set()
        patterns = []
        stop_words = self._get_intent_stop_words(intent)
        for utterance in intent_utterances:
            pattern = self._utterance_to_pattern(
                utterance, stop_words, entity_placeholders)
            if pattern not in unique_patterns:
                unique_patterns.add(pattern)
                patterns.append(pattern)
        return patterns

    def _utterance_to_pattern(self, utterance, stop_words,
                              entity_placeholders):
        slot_names_count = defaultdict(int)
        pattern = []
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name = chunk[SLOT_NAME]
                slot_names_count[slot_name] += 1
                group_name = self.slot_names_to_group_names[slot_name]
                count = slot_names_count[slot_name]
                if count > 1:
                    group_name = "%s_%s" % (group_name, count)
                placeholder = entity_placeholders[chunk[ENTITY]]
                pattern.append(r"(?P<%s>%s)" % (group_name, placeholder))
            else:
                tokens = tokenize_light(chunk[TEXT], self.language)
                pattern += [regex_escape(t.lower()) for t in tokens
                            if normalize(t) not in stop_words]

        pattern = r"^%s%s%s$" % (WHITESPACE_PATTERN,
                                 WHITESPACE_PATTERN.join(pattern),
                                 WHITESPACE_PATTERN)
        return pattern

    @check_persisted_path
    def persist(self, path):
        """Persists the object at the given path"""
        path.mkdir()
        parser_json = json_string(self.to_dict())
        parser_path = path / "intent_parser.json"

        with parser_path.open(mode="w") as f:
            f.write(parser_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`DeterministicIntentParser` instance from a path

        The data at the given path must have been generated using
        :func:`~DeterministicIntentParser.persist`
        """
        path = Path(path)
        model_path = path / "intent_parser.json"
        if not model_path.exists():
            raise LoadingError(
                "Missing deterministic intent parser metadata file: %s"
                % model_path.name)

        with model_path.open(encoding="utf8") as f:
            metadata = json.load(f)
        return cls.from_dict(metadata, **shared)

    def to_dict(self):
        """Returns a json-serializable dict"""
        stop_words_whitelist = None
        if self._stop_words_whitelist is not None:
            stop_words_whitelist = {
                intent: sorted(values)
                for intent, values in iteritems(self._stop_words_whitelist)}
        return {
            "config": self.config.to_dict(),
            "language_code": self.language,
            "patterns": self.patterns,
            "group_names_to_slot_names": self.group_names_to_slot_names,
            "slot_names_to_entities": self.slot_names_to_entities,
            "stop_words_whitelist": stop_words_whitelist
        }

    @classmethod
    def from_dict(cls, unit_dict, **shared):
        """Creates a :class:`DeterministicIntentParser` instance from a dict

        The dict must have been generated with
        :func:`~DeterministicIntentParser.to_dict`
        """
        config = cls.config_type.from_dict(unit_dict["config"])
        parser = cls(config=config, **shared)
        parser.patterns = unit_dict["patterns"]
        parser.language = unit_dict["language_code"]
        parser.group_names_to_slot_names = unit_dict[
            "group_names_to_slot_names"]
        parser.slot_names_to_entities = unit_dict["slot_names_to_entities"]
        if parser.fitted:
            whitelist = unit_dict.get("stop_words_whitelist", dict())
            # pylint:disable=protected-access
            parser._stop_words_whitelist = {
                intent: set(values) for intent, values in iteritems(whitelist)}
            # pylint:enable=protected-access
        return parser


def _get_range_shift(matched_range, ranges_mapping):
    shift = 0
    previous_replaced_range_end = None
    matched_start = matched_range[0]
    for replaced_range, orig_range in iteritems(ranges_mapping):
        if replaced_range[1] <= matched_start:
            if previous_replaced_range_end is None \
                    or replaced_range[1] > previous_replaced_range_end:
                previous_replaced_range_end = replaced_range[1]
                shift = orig_range[END] - replaced_range[1]
    return shift


def _get_group_names_to_slot_names(slot_names_mapping):
    slot_names = {slot_name for mapping in itervalues(slot_names_mapping)
                  for slot_name in iterkeys(mapping)}
    return {"group%s" % i: name
            for i, name in enumerate(sorted(slot_names))}


def _get_entity_placeholders(dataset, language):
    return {
        e: _get_entity_name_placeholder(e, language)
        for e in dataset[ENTITIES]
    }


def _deduplicate_overlapping_slots(slots, language):
    def overlap(lhs_slot, rhs_slot):
        return ranges_overlap(lhs_slot[RES_MATCH_RANGE],
                              rhs_slot[RES_MATCH_RANGE])

    def sort_key_fn(slot):
        tokens = tokenize(slot[RES_VALUE], language)
        return -(len(tokens) + len(slot[RES_VALUE]))

    deduplicated_slots = deduplicate_overlapping_items(
        slots, overlap, sort_key_fn)
    return sorted(deduplicated_slots,
                  key=lambda slot: slot[RES_MATCH_RANGE][START])


def _get_entity_name_placeholder(entity_label, language):
    return "%%%s%%" % "".join(
        tokenize_light(entity_label, language)).upper()


def _get_stop_words_whitelist(dataset, stop_words):
    entity_values_per_intent = extract_entity_values(
        dataset, apply_normalization=True)
    stop_words_whitelist = dict()
    for intent, entity_values in iteritems(entity_values_per_intent):
        whitelist = stop_words.intersection(entity_values)
        if whitelist:
            stop_words_whitelist[intent] = whitelist
    return stop_words_whitelist
