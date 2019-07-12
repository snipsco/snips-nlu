from __future__ import unicode_literals

import json
import logging
from builtins import str
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from future.utils import iteritems, itervalues
from snips_nlu_utils import normalize, hash_str

from snips_nlu.common.log_utils import log_elapsed_time, log_result
from snips_nlu.common.utils import (
    check_persisted_path, deduplicate_overlapping_entities, fitted_required,
    json_string)
from snips_nlu.constants import (
    DATA, END, ENTITIES, ENTITY, ENTITY_KIND, INTENTS, LANGUAGE, RES_INTENT,
    RES_INTENT_NAME, RES_MATCH_RANGE, RES_SLOTS, SLOT_NAME, START, TEXT,
    UTTERANCES, RES_PROBA)
from snips_nlu.dataset import (
    validate_and_format_dataset, extract_intent_entities)
from snips_nlu.dataset.utils import get_stop_words_whitelist
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.exceptions import IntentNotFoundError, LoadingError
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.pipeline.configs import LookupIntentParserConfig
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.resources import get_stop_words
from snips_nlu.result import (
    empty_result, intent_classification_result, parsing_result,
    unresolved_slot, extraction_result)

logger = logging.getLogger(__name__)


@IntentParser.register("lookup_intent_parser")
class LookupIntentParser(IntentParser):
    """A deterministic Intent parser implementation based on a dictionary

    This intent parser is very strict by nature, and tends to have a very good
    precision but a low recall. For this reason, it is interesting to use it
    first before potentially falling back to another parser.
    """

    config_type = LookupIntentParserConfig

    def __init__(self, config=None, **shared):
        """The lookup intent parser can be configured by passing a
        :class:`.LookupIntentParserConfig`"""
        super(LookupIntentParser, self).__init__(config, **shared)
        self._language = None
        self._stop_words = None
        self._stop_words_whitelist = None
        self._map = None
        self._intents_names = []
        self._slots_names = []
        self._intents_mapping = dict()
        self._slots_mapping = dict()
        self._entity_scopes = None

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
    def fitted(self):
        """Whether or not the intent parser has already been trained"""
        return self._map is not None

    @log_elapsed_time(
        logger, logging.INFO, "Fitted lookup intent parser in {elapsed_time}")
    def fit(self, dataset, force_retrain=True):
        """Fits the intent parser with a valid Snips dataset"""
        logger.info("Fitting lookup intent parser...")
        dataset = validate_and_format_dataset(dataset)
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        self.language = dataset[LANGUAGE]
        self._entity_scopes = _get_entity_scopes(dataset)
        self._map = dict()
        self._stop_words_whitelist = get_stop_words_whitelist(
            dataset, self._stop_words)
        entity_placeholders = _get_entity_placeholders(dataset, self.language)

        ambiguous_keys = set()
        for (key, val) in self._generate_io_mapping(dataset[INTENTS],
                                                    entity_placeholders):
            key = hash_str(key)
            # handle key collisions -*- flag ambiguous entries -*-
            if key in self._map and self._map[key] != val:
                ambiguous_keys.add(key)
            else:
                self._map[key] = val

        # delete ambiguous keys
        for key in ambiguous_keys:
            self._map.pop(key)

        return self

    @log_result(logger, logging.DEBUG, "LookupIntentParser result -> {result}")
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
                if intent[RES_PROBA] <= 0.5:
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

        results_per_intent = defaultdict(list)
        for text_candidate, entities in self._get_candidates(text, intents):
            val = self._map.get(hash_str(text_candidate))
            if val is not None:
                result = self._parse_map_output(text, val, entities, intents)
                if result:
                    intent_name = result[RES_INTENT][RES_INTENT_NAME]
                    results_per_intent[intent_name].append(result)

        results = []
        for intent_results in itervalues(results_per_intent):
            sorted_results = sorted(intent_results,
                                    key=lambda res: len(res[RES_SLOTS]))
            results.append(sorted_results[0])

        # In some rare cases there can be multiple ambiguous intents
        # In such cases, priority is given to results containing fewer slots
        weights = [1.0 / (1.0 + len(res[RES_SLOTS])) for res in results]
        total_weight = sum(weights)

        for res, weight in zip(results, weights):
            res[RES_INTENT][RES_PROBA] = weight / total_weight

        results = sorted(results, key=lambda r: -r[RES_INTENT][RES_PROBA])
        return results[:top_n]

    def _get_candidates(self, text, intents):
        candidates = defaultdict(list)
        for grouped_entity_scope in self._entity_scopes:
            entity_scope = grouped_entity_scope["entity_scope"]
            intent_group = grouped_entity_scope["intent_group"]
            intent_group = [intent_ for intent_ in intent_group
                            if intents is None or intent_ in intents]
            if not intent_group:
                continue

            builtin_entities = self.builtin_entity_parser.parse(
                text, scope=entity_scope["builtin"], use_cache=True)
            custom_entities = self.custom_entity_parser.parse(
                text, scope=entity_scope["custom"], use_cache=True)
            all_entities = builtin_entities + custom_entities
            all_entities = deduplicate_overlapping_entities(all_entities)

            # We generate all subsets of entities to match utterances
            # containing ambivalent words which can be both entity values or
            # random words
            for entities in _get_entities_combinations(all_entities):
                processed_text = self._replace_entities_with_placeholders(
                    text, entities)
                for intent in intent_group:
                    cleaned_text = self._preprocess_text(text, intent)
                    cleaned_processed_text = self._preprocess_text(
                        processed_text, intent)

                    raw_candidate = cleaned_text, []
                    placeholder_candidate = cleaned_processed_text, entities
                    intent_candidates = [raw_candidate, placeholder_candidate]
                    for text_input, text_entities in intent_candidates:
                        if text_input not in candidates \
                                or text_entities not in candidates[text_input]:
                            candidates[text_input].append(text_entities)
                            yield text_input, text_entities

    def _parse_map_output(self, text, output, entities, intents):
        """Parse the map output to the parser's result format"""
        intent_id, slot_ids = output
        intent_name = self._intents_names[intent_id]
        if intents is not None and intent_name not in intents:
            return None

        parsed_intent = intent_classification_result(
            intent_name=intent_name, probability=1.0)
        slots = []
        # assert invariant
        assert len(slot_ids) == len(entities)
        for slot_id, entity in zip(slot_ids, entities):
            slot_name = self._slots_names[slot_id]
            rng_start = entity[RES_MATCH_RANGE][START]
            rng_end = entity[RES_MATCH_RANGE][END]
            slot_value = text[rng_start:rng_end]
            entity_name = entity[ENTITY_KIND]
            slot = unresolved_slot(
                [rng_start, rng_end], slot_value, entity_name, slot_name)
            slots.append(slot)

        return extraction_result(parsed_intent, slots)

    @fitted_required
    def get_intents(self, text):
        """Returns the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent
        """
        nb_intents = len(self._intents_names)
        top_intents = [intent_result[RES_INTENT] for intent_result in
                       self._parse_top_intents(text, top_n=nb_intents)]
        matched_intents = {res[RES_INTENT_NAME] for res in top_intents}
        for intent in self._intents_names:
            if intent not in matched_intents:
                top_intents.append(intent_classification_result(intent, 0.0))

        # The None intent is not included in the lookup table and is thus
        # never matched by the lookup parser
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

        if intent not in self._intents_names:
            raise IntentNotFoundError(intent)

        slots = self.parse(text, intents=[intent])[RES_SLOTS]
        if slots is None:
            slots = []
        return slots

    def _get_intent_stop_words(self, intent):
        whitelist = self._stop_words_whitelist.get(intent, set())
        return self._stop_words.difference(whitelist)

    def _get_intent_id(self, intent_name):
        """generate a numeric id for an intent

        Args:
            intent_name (str): intent name

        Returns:
            int: numeric id

        """
        intent_id = self._intents_mapping.get(intent_name)
        if intent_id is None:
            intent_id = len(self._intents_names)
            self._intents_names.append(intent_name)
            self._intents_mapping[intent_name] = intent_id

        return intent_id

    def _get_slot_id(self, slot_name):
        """generate a numeric id for a slot

        Args:
            slot_name (str): intent name

        Returns:
            int: numeric id

        """
        slot_id = self._slots_mapping.get(slot_name)
        if slot_id is None:
            slot_id = len(self._slots_names)
            self._slots_names.append(slot_name)
            self._slots_mapping[slot_name] = slot_id

        return slot_id

    def _preprocess_text(self, txt, intent):
        """Replaces stop words and characters that are tokenized out by
            whitespaces"""
        stop_words = self._get_intent_stop_words(intent)
        tokens = tokenize_light(txt, self.language)
        cleaned_string = " ".join(
            [tkn for tkn in tokens if normalize(tkn) not in stop_words])
        return cleaned_string.lower()

    def _generate_io_mapping(self, intents, entity_placeholders):
        """Generate input-output pairs"""
        for intent_name, intent in sorted(iteritems(intents)):
            intent_id = self._get_intent_id(intent_name)
            for entry in intent[UTTERANCES]:
                yield self._build_io_mapping(
                    intent_id, entry, entity_placeholders)

    def _build_io_mapping(self, intent_id, utterance, entity_placeholders):
        input_ = []
        output = [intent_id]
        slots = []
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name = chunk[SLOT_NAME]
                slot_id = self._get_slot_id(slot_name)
                entity_name = chunk[ENTITY]
                placeholder = entity_placeholders[entity_name]
                input_.append(placeholder)
                slots.append(slot_id)
            else:
                input_.append(chunk[TEXT])
        output.append(slots)

        intent = self._intents_names[intent_id]
        key = self._preprocess_text(" ".join(input_), intent)

        return key, output

    def _replace_entities_with_placeholders(self, text, entities):
        if not entities:
            return text
        entities = sorted(entities, key=lambda e: e[RES_MATCH_RANGE][START])
        processed_text = ""
        current_idx = 0
        for ent in entities:
            start = ent[RES_MATCH_RANGE][START]
            end = ent[RES_MATCH_RANGE][END]
            processed_text += text[current_idx:start]
            place_holder = _get_entity_name_placeholder(
                ent[ENTITY_KIND], self.language)
            processed_text += place_holder
            current_idx = end
        processed_text += text[current_idx:]

        return processed_text

    @check_persisted_path
    def persist(self, path):
        """Persists the object at the given path"""
        path.mkdir()
        parser_json = json_string(self.to_dict())
        parser_path = path / "intent_parser.json"

        with parser_path.open(mode="w", encoding="utf8") as pfile:
            pfile.write(parser_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`LookupIntentParser` instance from a path

        The data at the given path must have been generated using
        :func:`~LookupIntentParser.persist`
        """
        path = Path(path)
        model_path = path / "intent_parser.json"
        if not model_path.exists():
            raise LoadingError(
                "Missing lookup intent parser metadata file: %s"
                % model_path.name)

        with model_path.open(encoding="utf8") as pfile:
            metadata = json.load(pfile)
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
            "map": self._map,
            "slots_names": self._slots_names,
            "intents_names": self._intents_names,
            "entity_scopes": self._entity_scopes,
            "stop_words_whitelist": stop_words_whitelist,
        }

    @classmethod
    def from_dict(cls, unit_dict, **shared):
        """Creates a :class:`LookupIntentParser` instance from a dict

        The dict must have been generated with
        :func:`~LookupIntentParser.to_dict`
        """
        config = cls.config_type.from_dict(unit_dict["config"])
        parser = cls(config=config, **shared)
        parser.language = unit_dict["language_code"]
        # pylint:disable=protected-access
        parser._map = _convert_dict_keys_to_int(unit_dict["map"])
        parser._slots_names = unit_dict["slots_names"]
        parser._intents_names = unit_dict["intents_names"]
        parser._entity_scopes = unit_dict["entity_scopes"]
        if parser.fitted:
            whitelist = unit_dict["stop_words_whitelist"]
            parser._stop_words_whitelist = {
                intent: set(values) for intent, values in iteritems(whitelist)}
        # pylint:enable=protected-access
        return parser


def _get_entity_scopes(dataset):
    intent_entities = extract_intent_entities(dataset)
    intent_groups = []
    entity_scopes = []
    for intent, entities in sorted(iteritems(intent_entities)):
        scope = {
            "builtin": list(
                {ent for ent in entities if is_builtin_entity(ent)}),
            "custom": list(
                {ent for ent in entities if not is_builtin_entity(ent)})
        }
        if scope in entity_scopes:
            group_idx = entity_scopes.index(scope)
            intent_groups[group_idx].append(intent)
        else:
            entity_scopes.append(scope)
            intent_groups.append([intent])
    return [
        {
            "intent_group": intent_group,
            "entity_scope": entity_scope
        } for intent_group, entity_scope in zip(intent_groups, entity_scopes)
    ]


def _get_entity_placeholders(dataset, language):
    return {
        e: _get_entity_name_placeholder(e, language) for e in dataset[ENTITIES]
    }


def _get_entity_name_placeholder(entity_label, language):
    return "%%%s%%" % "".join(tokenize_light(entity_label, language)).upper()


def _convert_dict_keys_to_int(dct):
    if isinstance(dct, dict):
        return {int(k): v for k, v in iteritems(dct)}
    return dct


def _get_entities_combinations(entities):
    yield ()
    for nb_entities in reversed(range(1, len(entities) + 1)):
        for combination in combinations(entities, nb_entities):
            yield combination
