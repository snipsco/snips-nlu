from __future__ import unicode_literals

import json
import logging
from builtins import str
from pathlib import Path

from future.utils import iteritems

from snips_nlu_utils import normalize

from snips_nlu.common.log_utils import log_elapsed_time, log_result
from snips_nlu.common.utils import (
    check_persisted_path,
    deduplicate_overlapping_entities,
    fitted_required,
    json_string,
)
from snips_nlu.constants import (
    DATA,
    END,
    ENTITIES,
    ENTITY,
    ENTITY_KIND,
    INTENTS,
    LANGUAGE,
    RES_INPUT,
    RES_INTENT,
    RES_INTENT_NAME,
    RES_MATCH_RANGE,
    RES_SLOTS,
    SLOT_NAME,
    START,
    TEXT,
    UTTERANCES,
)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.exceptions import IntentNotFoundError, LoadingError
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.pipeline.configs import LookupIntentParserConfig
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.resources import get_stop_words
from snips_nlu.result import (
    empty_result,
    intent_classification_result,
    parsing_result,
    unresolved_slot,
)

logger = logging.getLogger(__name__)


@IntentParser.register("lookup_intent_parser")
class LookupIntentParser(IntentParser):
    """A Deterministic Intent parser implementation based on a dictionary

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
        self.stop_words = None
        self.map = None
        self.intents_names = []
        self.slots_names = []
        self._intents_mapping = dict()
        self._slots_mapping = dict()

    @property
    def language(self):
        """get parser's language"""
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        if value is None:
            self.stop_words = None
        else:
            if self.config.ignore_stop_words:
                self.stop_words = get_stop_words(self.resources)
            else:
                self.stop_words = set()

    @property
    def fitted(self):
        """Whether or not the intent parser has already been trained"""
        return self.map is not None

    @log_elapsed_time(
        logger, logging.INFO, "Fitted lookup intent parser in {elapsed_time}"
    )
    def fit(self, dataset, force_retrain=True):
        """Fits the intent parser with a valid Snips dataset"""
        logger.info("Fitting lookup intent parser...")
        dataset = validate_and_format_dataset(dataset)
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        self.language = dataset[LANGUAGE]
        self.map = dict()
        entity_placeholders = _get_entity_placeholders(dataset, self.language)

        ambiguous_keys = set()
        for (key, val) in self.generate_io_mapping(
                dataset[INTENTS], entity_placeholders
        ):
            # handle key collisions -*- remove ambiguous entries -*-
            if key in self.map and self.map[key] != val:
                ambiguous_keys.add(key)
            else:
                self.map[key] = val

        # delete ambiguous keys
        for key in ambiguous_keys:
            self.map.pop(key)

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
        if isinstance(intents, str):
            intents = [intents]

        builtin_entities = self.builtin_entity_parser.parse(
            text, use_cache=True
        )
        custom_entities = self.custom_entity_parser.parse(text, use_cache=True)
        all_entities = builtin_entities + custom_entities

        all_entities = deduplicate_overlapping_entities(all_entities)

        processed_text = self._replace_entities_with_placeholders(
            text, all_entities
        )

        cleaned_processed_text = self._preprocess_text(processed_text)
        cleaned_text = self._preprocess_text(text)

        val = self.map.get(cleaned_processed_text)

        if val is None:
            val = self.map.get(cleaned_text)
            all_entities = []

        # conform to api
        result = self.parse_map_output(text, val, all_entities, intents)
        if top_n is not None:
            # convert parsing_result to extraction_result and return a list
            result.pop(RES_INPUT)
            result = [result]

        return result

    def parse_map_output(self, text, output, entities, intents):
        """Parse the trie output to the parser's result format"""
        if not output:
            return empty_result(text, 1.0)
        intent_id, slot_ids = output
        intent_name = self.intents_names[intent_id]
        if intents is not None and intent_name not in intents:
            return empty_result(text, 1.0)

        parsed_intent = intent_classification_result(
            intent_name=intent_name, probability=1.0
        )
        slots = []
        # assert invariant
        assert len(slot_ids) == len(entities)
        for slot_id, entity in zip(slot_ids, entities):
            slot_name = self.slots_names[slot_id]
            slot_value = text[
                entity[RES_MATCH_RANGE][START] : entity[RES_MATCH_RANGE][END]
            ]
            entity_name = entity[ENTITY_KIND]
            start_end = [
                entity[RES_MATCH_RANGE][START],
                entity[RES_MATCH_RANGE][END],
            ]
            slot = unresolved_slot(
                start_end, slot_value, entity_name, slot_name
            )
            slots.append(slot)

        return parsing_result(text, parsed_intent, slots)

    @fitted_required
    def get_intents(self, text):
        """Returns the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent
        """
        intents = []
        res = self.parse(text)

        matched_intent = res[RES_INTENT]
        intent_name = matched_intent[RES_INTENT_NAME]
        intents.append(matched_intent)
        others = [x for x in self.intents_names if x != intent_name]

        for intent in others:
            intents.append(intent_classification_result(intent, 0.0))

        if intent_name is not None:
            intents.append(intent_classification_result(None, 0.0))

        return intents

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

        if intent not in self.intents_names:
            raise IntentNotFoundError(intent)

        slots = self.parse(text, intents=[intent])[RES_SLOTS]
        if slots is None:
            slots = []
        return slots

    def get_intent_id(self, intent_name):
        """generate a numeric id for an intent

        Args:
            intent_name (str): intent name

        Returns:
            int: numeric id

        """
        intent_id = self._intents_mapping.get(intent_name)
        if intent_id is None:
            intent_id = len(self.intents_names)
            self.intents_names.append(intent_name)
            self._intents_mapping[intent_name] = intent_id

        return intent_id

    def get_slot_id(self, slot_name):
        """generate a numeric id for a slot

        Args:
            slot_name (str): intent name

        Returns:
            int: numeric id

        """
        slot_id = self._slots_mapping.get(slot_name)
        if slot_id is None:
            slot_id = len(self.slots_names)
            self.slots_names.append(slot_name)
            self._slots_mapping[slot_name] = slot_id

        return slot_id

    def _preprocess_text(self, txt):
        """Replaces stop words and characters that are tokenized out by
            whitespaces"""
        tokens = tokenize_light(txt, self.language)
        cleaned_string = " ".join(
            [tkn for tkn in tokens if normalize(tkn) not in self.stop_words]
        )
        return cleaned_string.lower()

    def generate_io_mapping(self, intents, entity_placeholders):
        """Generate input-output pairs"""
        for intent_name, intent in iteritems(intents):
            intent_id = self.get_intent_id(intent_name)
            for entry in intent[UTTERANCES]:
                yield self._build_io_mapping(
                    intent_id, entry, entity_placeholders
                )

    def _build_io_mapping(self, intent_id, utterance, entity_placeholders):
        input_ = []
        output = [intent_id]
        slots = []
        for chunk in utterance[DATA]:
            if SLOT_NAME in chunk:
                slot_name = chunk[SLOT_NAME]
                slot_id = self.get_slot_id(slot_name)
                entity_name = chunk[ENTITY]
                placeholder = entity_placeholders[entity_name]
                input_.append(placeholder.lower())
                slots.append(slot_id)
            else:
                input_.append(chunk[TEXT])
        output.append(slots)

        key = self._preprocess_text(" ".join(input_))

        return key, output

    def _replace_entities_with_placeholders(self, text, entities):
        if not entities:
            return text
        entities.sort(key=lambda e: e[RES_MATCH_RANGE][START])
        processed_text = ""
        current_idx = 0
        for ent in entities:
            start = ent[RES_MATCH_RANGE][START]
            end = ent[RES_MATCH_RANGE][END]
            processed_text += text[current_idx:start]
            place_holder = _get_entity_name_placeholder(
                ent[ENTITY_KIND], self.language
            )
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

        with parser_path.open(mode="w") as pfile:
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
                % model_path.name
            )

        with model_path.open(encoding="utf8") as pfile:
            metadata = json.load(pfile)
        return cls.from_dict(metadata, **shared)

    def to_dict(self):
        """Returns a json-serializable dict"""
        return {
            "config": self.config.to_dict(),
            "language_code": self.language,
            "map": self.map,
            "slots_names": self.slots_names,
            "intents_names": self.intents_names,
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
        parser.map = unit_dict["map"]
        parser.slots_names = unit_dict["slots_names"]
        parser.intents_names = unit_dict["intents_names"]

        return parser


def _get_entity_placeholders(dataset, language):
    return {
        e: _get_entity_name_placeholder(e, language) for e in dataset[ENTITIES]
    }


def _get_entity_name_placeholder(entity_label, language):
    return "%%%s%%" % "".join(tokenize_light(entity_label, language)).upper()
