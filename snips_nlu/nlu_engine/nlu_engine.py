from __future__ import unicode_literals

import json
import logging
from builtins import str
from collections import defaultdict
from pathlib import Path

from future.utils import iteritems, itervalues

from snips_nlu.__about__ import __model_version__, __version__
from snips_nlu.common.dataset_utils import get_slot_name_mappings
from snips_nlu.common.log_utils import log_elapsed_time
from snips_nlu.common.utils import (
    check_persisted_path, fitted_required, json_string)
from snips_nlu.constants import (
    AUTOMATICALLY_EXTENSIBLE, BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER,
    ENTITIES, ENTITY_KIND, LANGUAGE, RESOLVED_VALUE, RES_ENTITY,
    RES_INTENT, RES_INTENT_NAME, RES_MATCH_RANGE, RES_PROBA, RES_SLOTS,
    RES_VALUE, RESOURCES)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.default_configs import DEFAULT_CONFIGS
from snips_nlu.entity_parser import CustomEntityParser
from snips_nlu.entity_parser.builtin_entity_parser import (
    BuiltinEntityParser, is_builtin_entity)
from snips_nlu.exceptions import InvalidInputError, IntentNotFoundError, \
    LoadingError, IncompatibleModelError
from snips_nlu.intent_parser import IntentParser
from snips_nlu.pipeline.configs import NLUEngineConfig
from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.resources import load_resources_from_dir, persist_resources
from snips_nlu.result import (
    builtin_slot, custom_slot, empty_result, extraction_result, is_empty,
    parsing_result)

logger = logging.getLogger(__name__)


@ProcessingUnit.register("nlu_engine")
class SnipsNLUEngine(ProcessingUnit):
    """Main class to use for intent parsing

    A :class:`SnipsNLUEngine` relies on a list of :class:`.IntentParser`
    object to parse intents, by calling them successively using the first
    positive output.

    With the default parameters, it will use the two following intent parsers
    in this order:

    - a :class:`.DeterministicIntentParser`
    - a :class:`.ProbabilisticIntentParser`

    The logic behind is to first use a conservative parser which has a very
    good precision while its recall is modest, so simple patterns will be
    caught, and then fallback on a second parser which is machine-learning
    based and will be able to parse unseen utterances while ensuring a good
    precision and recall.
    """

    config_type = NLUEngineConfig

    def __init__(self, config=None, **shared):
        """The NLU engine can be configured by passing a
        :class:`.NLUEngineConfig`"""
        super(SnipsNLUEngine, self).__init__(config, **shared)
        self.intent_parsers = []
        """list of :class:`.IntentParser`"""
        self.dataset_metadata = None

    @classmethod
    def default_config(cls):
        # Do not use the global default config, and use per-language default
        # configs instead
        return None

    @property
    def fitted(self):
        """Whether or not the nlu engine has already been fitted"""
        return self.dataset_metadata is not None

    @log_elapsed_time(
        logger, logging.INFO, "Fitted NLU engine in {elapsed_time}")
    def fit(self, dataset, force_retrain=True):
        """Fits the NLU engine

        Args:
            dataset (dict): A valid Snips dataset
            force_retrain (bool, optional): If *False*, will not retrain intent
                parsers when they are already fitted. Default to *True*.

        Returns:
            The same object, trained.
        """
        dataset = validate_and_format_dataset(dataset)
        if self.config is None:
            language = dataset[LANGUAGE]
            default_config = DEFAULT_CONFIGS.get(language)
            if default_config is not None:
                self.config = self.config_type.from_dict(default_config)
            else:
                self.config = self.config_type()

        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        parsers = []
        for parser_config in self.config.intent_parsers_configs:
            # Re-use existing parsers to allow pre-training
            recycled_parser = None
            for parser in self.intent_parsers:
                if parser.unit_name == parser_config.unit_name:
                    recycled_parser = parser
                    break
            if recycled_parser is None:
                recycled_parser = IntentParser.from_config(
                    parser_config,
                    builtin_entity_parser=self.builtin_entity_parser,
                    custom_entity_parser=self.custom_entity_parser,
                    resources=self.resources)

            if force_retrain or not recycled_parser.fitted:
                recycled_parser.fit(dataset, force_retrain)
            parsers.append(recycled_parser)

        self.intent_parsers = parsers
        self.dataset_metadata = _get_dataset_metadata(dataset)
        return self

    @log_elapsed_time(logger, logging.DEBUG, "Parsed input in {elapsed_time}")
    @fitted_required
    def parse(self, text, intents=None, top_n=None):
        """Performs intent parsing on the provided *text* by calling its intent
        parsers successively

        Args:
            text (str): Input
            intents (str or list of str, optional): If provided, reduces the
                scope of intent parsing to the provided list of intents
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
            NotTrained: When the nlu engine is not fitted
            InvalidInputError: When input type is not unicode
        """
        if not isinstance(text, str):
            raise InvalidInputError("Expected unicode but received: %s"
                                    % type(text))

        if isinstance(intents, str):
            intents = {intents}
        elif isinstance(intents, list):
            intents = set(intents)

        if intents is not None:
            for intent in intents:
                if intent not in self.dataset_metadata["slot_name_mappings"]:
                    raise IntentNotFoundError(intent)

        if top_n is None:
            none_proba = 0.0
            for parser in self.intent_parsers:
                res = parser.parse(text, intents)
                if is_empty(res):
                    none_proba = res[RES_INTENT][RES_PROBA]
                    continue
                resolved_slots = self._resolve_slots(text, res[RES_SLOTS])
                return parsing_result(text, intent=res[RES_INTENT],
                                      slots=resolved_slots)
            return empty_result(text, none_proba)

        intents_results = self.get_intents(text)
        if intents is not None:
            intents_results = [res for res in intents_results
                               if res[RES_INTENT_NAME] is None
                               or res[RES_INTENT_NAME] in intents]
        intents_results = intents_results[:top_n]
        results = []
        for intent_res in intents_results:
            slots = self.get_slots(text, intent_res[RES_INTENT_NAME])
            results.append(extraction_result(intent_res, slots))
        return results

    @log_elapsed_time(logger, logging.DEBUG, "Got intents in {elapsed_time}")
    @fitted_required
    def get_intents(self, text):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent

        .. note::

            The probabilities returned along with each intent are not
            guaranteed to sum to 1.0. They should be considered as scores
            between 0 and 1.
        """
        results = None
        for parser in self.intent_parsers:
            parser_results = parser.get_intents(text)
            if results is None:
                results = {res[RES_INTENT_NAME]: res for res in parser_results}
                continue

            for res in parser_results:
                intent = res[RES_INTENT_NAME]
                proba = max(res[RES_PROBA], results[intent][RES_PROBA])
                results[intent][RES_PROBA] = proba

        return sorted(itervalues(results), key=lambda res: -res[RES_PROBA])

    @log_elapsed_time(logger, logging.DEBUG, "Parsed slots in {elapsed_time}")
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
            InvalidInputError: When input type is not unicode
        """
        if not isinstance(text, str):
            raise InvalidInputError("Expected unicode but received: %s"
                                    % type(text))

        if intent is None:
            return []

        if intent not in self.dataset_metadata["slot_name_mappings"]:
            raise IntentNotFoundError(intent)

        for parser in self.intent_parsers:
            slots = parser.get_slots(text, intent)
            if not slots:
                continue
            return self._resolve_slots(text, slots)
        return []

    @check_persisted_path
    def persist(self, path):
        """Persists the NLU engine at the given directory path

        Args:
            path (str or pathlib.Path): the location at which the nlu engine
                must be persisted. This path must not exist when calling this
                function.

        Raises:
            PersistingError: when persisting to a path which already exists
        """
        path.mkdir()

        parsers_count = defaultdict(int)
        intent_parsers = []
        for parser in self.intent_parsers:
            parser_name = parser.unit_name
            parsers_count[parser_name] += 1
            count = parsers_count[parser_name]
            if count > 1:
                parser_name = "{n}_{c}".format(n=parser_name, c=count)
            parser_path = path / parser_name
            parser.persist(parser_path)
            intent_parsers.append(parser_name)

        config = None
        if self.config is not None:
            config = self.config.to_dict()

        builtin_entity_parser = None
        if self.builtin_entity_parser is not None:
            builtin_entity_parser = "builtin_entity_parser"
            builtin_entity_parser_path = path / builtin_entity_parser
            self.builtin_entity_parser.persist(builtin_entity_parser_path)

        custom_entity_parser = None
        if self.custom_entity_parser is not None:
            custom_entity_parser = "custom_entity_parser"
            custom_entity_parser_path = path / custom_entity_parser
            self.custom_entity_parser.persist(custom_entity_parser_path)

        model = {
            "unit_name": self.unit_name,
            "dataset_metadata": self.dataset_metadata,
            "intent_parsers": intent_parsers,
            "custom_entity_parser": custom_entity_parser,
            "builtin_entity_parser": builtin_entity_parser,
            "config": config,
            "model_version": __model_version__,
            "training_package_version": __version__
        }

        model_json = json_string(model)
        model_path = path / "nlu_engine.json"
        with model_path.open(mode="w") as f:
            f.write(model_json)

        if self.fitted:
            required_resources = self.config.get_required_resources()
            language = self.dataset_metadata["language_code"]
            resources_path = path / "resources"
            resources_path.mkdir()
            persist_resources(self.resources, resources_path / language,
                              required_resources)

    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`SnipsNLUEngine` instance from a directory path

        The data at the given path must have been generated using
        :func:`~SnipsNLUEngine.persist`

        Args:
            path (str): The path where the nlu engine is stored

        Raises:
            LoadingError: when some files are missing
            IncompatibleModelError: when trying to load an engine model which
                is not compatible with the current version of the lib
        """
        directory_path = Path(path)
        model_path = directory_path / "nlu_engine.json"
        if not model_path.exists():
            raise LoadingError("Missing nlu engine model file: %s"
                               % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model = json.load(f)
        model_version = model.get("model_version")
        if model_version is None or model_version != __model_version__:
            raise IncompatibleModelError(model_version)

        dataset_metadata = model["dataset_metadata"]
        if shared.get(RESOURCES) is None and dataset_metadata is not None:
            language = dataset_metadata["language_code"]
            resources_dir = directory_path / "resources" / language
            if resources_dir.is_dir():
                resources = load_resources_from_dir(resources_dir)
                shared[RESOURCES] = resources

        if shared.get(BUILTIN_ENTITY_PARSER) is None:
            path = model["builtin_entity_parser"]
            if path is not None:
                parser_path = directory_path / path
                shared[BUILTIN_ENTITY_PARSER] = BuiltinEntityParser.from_path(
                    parser_path)

        if shared.get(CUSTOM_ENTITY_PARSER) is None:
            path = model["custom_entity_parser"]
            if path is not None:
                parser_path = directory_path / path
                shared[CUSTOM_ENTITY_PARSER] = CustomEntityParser.from_path(
                    parser_path)

        config = cls.config_type.from_dict(model["config"])
        nlu_engine = cls(config=config, **shared)
        nlu_engine.dataset_metadata = dataset_metadata
        intent_parsers = []
        for parser_idx, parser_name in enumerate(model["intent_parsers"]):
            parser_config = config.intent_parsers_configs[parser_idx]
            intent_parser_path = directory_path / parser_name
            intent_parser = IntentParser.load_from_path(
                intent_parser_path, parser_config.unit_name, **shared)
            intent_parsers.append(intent_parser)
        nlu_engine.intent_parsers = intent_parsers
        return nlu_engine

    def _resolve_slots(self, text, slots):
        builtin_scope = [slot[RES_ENTITY] for slot in slots
                         if is_builtin_entity(slot[RES_ENTITY])]
        custom_scope = [slot[RES_ENTITY] for slot in slots
                        if not is_builtin_entity(slot[RES_ENTITY])]
        # Do not use cached entities here as datetimes must be computed using
        # current context
        builtin_entities = self.builtin_entity_parser.parse(
            text, builtin_scope, use_cache=False)
        custom_entities = self.custom_entity_parser.parse(
            text, custom_scope, use_cache=True)

        resolved_slots = []
        for slot in slots:
            entity_name = slot[RES_ENTITY]
            raw_value = slot[RES_VALUE]
            is_builtin = is_builtin_entity(entity_name)
            if is_builtin:
                entities = builtin_entities
                parser = self.builtin_entity_parser
                slot_builder = builtin_slot
                use_cache = False
                extensible = False
            else:
                entities = custom_entities
                parser = self.custom_entity_parser
                slot_builder = custom_slot
                use_cache = True
                extensible = self.dataset_metadata[ENTITIES][entity_name][
                    AUTOMATICALLY_EXTENSIBLE]

            resolved_slot = None
            for ent in entities:
                if ent[ENTITY_KIND] == entity_name and \
                        ent[RES_MATCH_RANGE] == slot[RES_MATCH_RANGE]:
                    resolved_slot = slot_builder(slot, ent[RESOLVED_VALUE])
                    break
            if resolved_slot is None:
                matches = parser.parse(
                    raw_value, scope=[entity_name], use_cache=use_cache)
                if matches:
                    match = matches[0]
                    if is_builtin or len(match[RES_VALUE]) == len(raw_value):
                        resolved_slot = slot_builder(
                            slot, match[RESOLVED_VALUE])

            if resolved_slot is None and extensible:
                resolved_slot = slot_builder(slot)

            if resolved_slot is not None:
                resolved_slots.append(resolved_slot)

        return resolved_slots


def _get_dataset_metadata(dataset):
    dataset = dataset
    entities = dict()
    for entity_name, entity in iteritems(dataset[ENTITIES]):
        if is_builtin_entity(entity_name):
            continue
        entities[entity_name] = {
            AUTOMATICALLY_EXTENSIBLE: entity[AUTOMATICALLY_EXTENSIBLE]
        }
    slot_name_mappings = get_slot_name_mappings(dataset)
    return {
        "language_code": dataset[LANGUAGE],
        "entities": entities,
        "slot_name_mappings": slot_name_mappings
    }
