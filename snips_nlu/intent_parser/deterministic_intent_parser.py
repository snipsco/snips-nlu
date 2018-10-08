from __future__ import unicode_literals

import json
import logging
import re
from builtins import str
from pathlib import Path

from future.utils import iteritems

from snips_nlu.constants import (
    BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER, DATA, END, ENTITIES, ENTITY,
    ENTITY_KIND, INTENTS, LANGUAGE, RES_MATCH_RANGE, RES_VALUE, SLOT_NAME,
    START, TEXT, UTTERANCES)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.pipeline.configs import DeterministicIntentParserConfig
from snips_nlu.preprocessing import tokenize, tokenize_light
from snips_nlu.result import (
    empty_result, intent_classification_result, parsing_result,
    unresolved_slot)
from snips_nlu.utils import (
    check_persisted_path, deduplicate_overlapping_items, fitted_required,
    get_slot_name_mappings, json_string, log_elapsed_time, log_result,
    ranges_overlap, regex_escape)

GROUP_NAME_PREFIX = "group"
GROUP_NAME_SEPARATOR = "_"
WHITESPACE_PATTERN = r"\s*"

logger = logging.getLogger(__name__)


class DeterministicIntentParser(IntentParser):
    """Intent parser using pattern matching in a deterministic manner

    This intent parser is very strict by nature, and tends to have a very good
    precision but a low recall. For this reason, it is interesting to use it
    first before potentially falling back to another parser.
    """

    unit_name = "deterministic_intent_parser"
    config_type = DeterministicIntentParserConfig

    def __init__(self, config=None, **shared):
        """The deterministic intent parser can be configured by passing a
        :class:`.DeterministicIntentParserConfig`"""
        if config is None:
            config = self.config_type()
        super(DeterministicIntentParser, self).__init__(config, **shared)
        self.language = None
        self.regexes_per_intent = None
        self.group_names_to_slot_names = None
        self.slot_names_to_entities = None

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
        """Fit the intent parser with a valid Snips dataset"""
        logger.info("Fitting deterministic parser...")
        dataset = validate_and_format_dataset(dataset)
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        self.language = dataset[LANGUAGE]
        self.regexes_per_intent = dict()
        self.group_names_to_slot_names = dict()
        entity_placeholders = _get_entity_placeholders(dataset, self.language)
        self.slot_names_to_entities = get_slot_name_mappings(dataset)
        for intent_name, intent in iteritems(dataset[INTENTS]):
            utterances = intent[UTTERANCES]
            patterns, self.group_names_to_slot_names = _generate_patterns(
                utterances, entity_placeholders,
                self.group_names_to_slot_names, self.language)
            patterns = [p for p in patterns
                        if len(p) < self.config.max_pattern_length]
            patterns = patterns[:self.config.max_queries]
            regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
            self.regexes_per_intent[intent_name] = regexes
        return self

    @log_result(
        logger, logging.DEBUG, "DeterministicIntentParser result -> {result}")
    @log_elapsed_time(logger, logging.DEBUG, "Parsed in {elapsed_time}.")
    @fitted_required
    def parse(self, text, intents=None):
        """Performs intent parsing on the provided *text*

        Intent and slots are extracted simultaneously through pattern matching

        Args:
            text (str): Input
            intents (str or list of str): If provided, reduces the scope of
            intent parsing to the provided list of intents

        Returns:
            dict: The matched intent, if any, along with the extracted slots.
            See :func:`.parsing_result` for the output format.

        Raises:
            NotTrained: When the intent parser is not fitted
        """
        logger.debug("DeterministicIntentParser parsing '%s'...", text)

        if isinstance(intents, str):
            intents = [intents]

        builtin_entities = self.builtin_entity_parser.parse(
            text, use_cache=True)
        custom_entities = self.custom_entity_parser.parse(
            text, use_cache=True)
        all_entities = builtin_entities + custom_entities
        ranges_mapping, processed_text = _replace_entities_with_placeholders(
            text, self.language, all_entities)

        # We try to match both the input text and the preprocessed text to
        # cover inconsistencies between labeled data and builtin entity parsing
        cleaned_text = _replace_tokenized_out_characters(text, self.language)
        cleaned_processed_text = _replace_tokenized_out_characters(
            processed_text, self.language)

        for intent, regexes in iteritems(self.regexes_per_intent):
            if intents is not None and intent not in intents:
                continue
            for regex in regexes:
                res = self._get_matching_result(text, cleaned_processed_text,
                                                regex, intent, ranges_mapping)
                if res is None:
                    res = self._get_matching_result(text, cleaned_text, regex,
                                                    intent)
                if res is not None:
                    return res
        return empty_result(text)

    def _get_matching_result(self, text, processed_text, regex, intent,
                             entities_ranges_mapping=None):
        found_result = regex.match(processed_text)
        if found_result is None:
            return None
        parsed_intent = intent_classification_result(intent_name=intent,
                                                     probability=1.0)
        slots = []
        for group_name in found_result.groupdict():
            slot_name = self.group_names_to_slot_names[group_name]
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
        return parsing_result(text, parsed_intent, parsed_slots)

    @check_persisted_path
    def persist(self, path):
        """Persist the object at the given path"""
        path = Path(path)
        path.mkdir()
        parser_json = json_string(self.to_dict())
        parser_path = path / "intent_parser.json"

        with parser_path.open(mode="w") as f:
            f.write(parser_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        """Load a :class:`DeterministicIntentParser` instance from a path

        The data at the given path must have been generated using
        :func:`~DeterministicIntentParser.persist`
        """
        path = Path(path)
        metadata_path = path / "intent_parser.json"
        if not metadata_path.exists():
            raise OSError("Missing deterministic intent parser metadata file: "
                          "%s" % metadata_path.name)

        with metadata_path.open(encoding="utf8") as f:
            metadata = json.load(f)
        return cls.from_dict(metadata, **shared)

    def to_dict(self):
        """Returns a json-serializable dict"""
        return {
            "config": self.config.to_dict(),
            "language_code": self.language,
            "patterns": self.patterns,
            "group_names_to_slot_names": self.group_names_to_slot_names,
            "slot_names_to_entities": self.slot_names_to_entities
        }

    @classmethod
    def from_dict(cls, unit_dict, **shared):
        """Creates a :class:`DeterministicIntentParser` instance from a dict

        The dict must have been generated with
        :func:`~DeterministicIntentParser.to_dict`
        """
        config = cls.config_type.from_dict(unit_dict["config"])
        parser = cls(
            config=config,
            builtin_entity_parser=shared.get(BUILTIN_ENTITY_PARSER),
            custom_entity_parser=shared.get(CUSTOM_ENTITY_PARSER),
        )
        parser.patterns = unit_dict["patterns"]
        parser.language = unit_dict["language_code"]
        parser.group_names_to_slot_names = unit_dict[
            "group_names_to_slot_names"]
        parser.slot_names_to_entities = unit_dict["slot_names_to_entities"]
        return parser


def _replace_tokenized_out_characters(string, language, replacement_char=" "):
    """Replace all characters that are tokenized out by `replacement_char`

    Examples:

        >>> string = "hello, it's me"
        >>> language = "en"
        >>> tokenize_light(string, language)
        ['hello', 'it', 's', 'me']
        >>> _replace_tokenized_out_characters(string, language, "_")
        'hello__it_s_me'
    """
    tokens = tokenize(string, language)
    current_idx = 0
    cleaned_string = ""
    for token in tokens:
        prefix_length = token.start - current_idx
        cleaned_string += "".join(
            (replacement_char for _ in range(prefix_length)))
        cleaned_string += token.value
        current_idx = token.end
    suffix_length = len(string) - current_idx
    cleaned_string += "".join((replacement_char for _ in range(suffix_length)))
    return cleaned_string


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


def _get_index(index):
    split = index.split(GROUP_NAME_SEPARATOR)
    if len(split) != 2 or split[0] != GROUP_NAME_PREFIX:
        raise ValueError("Misformatted index")
    return int(split[1])


def _make_index(i):
    return "%s%s%s" % (GROUP_NAME_PREFIX, GROUP_NAME_SEPARATOR, i)


def _generate_new_index(slots_name_to_labels):
    if not slots_name_to_labels:
        index = _make_index(0)
    else:
        max_index = max(slots_name_to_labels, key=_get_index)
        max_index = _get_index(max_index) + 1
        index = _make_index(max_index)
    return index


def _query_to_pattern(query, entity_placeholders, group_names_to_slot_names,
                      language):
    pattern = []
    for chunk in query[DATA]:
        if SLOT_NAME in chunk:
            max_index = _generate_new_index(group_names_to_slot_names)
            slot_name = chunk[SLOT_NAME]
            entity = chunk[ENTITY]
            group_names_to_slot_names[max_index] = slot_name
            pattern.append(
                r"(?P<%s>%s)" % (max_index, entity_placeholders[entity]))
        else:
            tokens = tokenize_light(chunk[TEXT], language)
            pattern += [regex_escape(t) for t in tokens]

    pattern = r"^%s%s%s$" % (WHITESPACE_PATTERN,
                             WHITESPACE_PATTERN.join(pattern),
                             WHITESPACE_PATTERN)
    return pattern, group_names_to_slot_names


def _get_queries_with_unique_context(intent_queries, language):
    contexts = set()
    queries = []
    for query in intent_queries:
        context = ""
        for chunk in query[DATA]:
            if ENTITY not in chunk:
                context += chunk[TEXT]
            else:
                context += _get_entity_name_placeholder(chunk[ENTITY],
                                                        language)
        context = context.strip()
        if context not in contexts:
            contexts.add(context)
            queries.append(query)
    return queries


def _generate_patterns(intent_queries, entity_placeholders,
                       group_names_to_labels, language):
    queries = _get_queries_with_unique_context(intent_queries, language)
    # Join all the entities utterances with a "|" to create the patterns
    patterns = set()
    for query in queries:
        pattern, group_names_to_labels = _query_to_pattern(
            query, entity_placeholders, group_names_to_labels, language)
        patterns.add(pattern)
    return list(patterns), group_names_to_labels


def _get_entity_placeholders(dataset, language):
    return {
        e: _get_entity_name_placeholder(e, language)
        for e in dataset[ENTITIES]
    }


def _replace_entities_with_placeholders(text, language, entities):
    if not entities:
        return dict(), text

    entities = _deduplicate_overlapping_entities(entities)
    entities = sorted(
        entities, key=lambda e: e[RES_MATCH_RANGE][START])

    range_mapping = dict()
    processed_text = ""
    offset = 0
    current_ix = 0
    for ent in entities:
        ent_start = ent[RES_MATCH_RANGE][START]
        ent_end = ent[RES_MATCH_RANGE][END]
        rng_start = ent_start + offset

        processed_text += text[current_ix:ent_start]

        entity_length = ent_end - ent_start
        entity_place_holder = _get_entity_name_placeholder(
            ent[ENTITY_KIND], language)

        offset += len(entity_place_holder) - entity_length

        processed_text += entity_place_holder
        rng_end = ent_end + offset
        new_range = (rng_start, rng_end)
        range_mapping[new_range] = ent[RES_MATCH_RANGE]
        current_ix = ent_end

    processed_text += text[current_ix:]
    return range_mapping, processed_text


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


def _deduplicate_overlapping_entities(entities):
    def overlap(lhs_entity, rhs_entity):
        return ranges_overlap(lhs_entity[RES_MATCH_RANGE],
                              rhs_entity[RES_MATCH_RANGE])

    def sort_key_fn(entity):
        return -len(entity[RES_VALUE])

    deduplicated_entities = deduplicate_overlapping_items(
        entities, overlap, sort_key_fn)
    return sorted(deduplicated_entities,
                  key=lambda entity: entity[RES_MATCH_RANGE][START])


def _get_entity_name_placeholder(entity_label, language):
    return "%%%s%%" % "".join(
        tokenize_light(entity_label, language)).upper()
