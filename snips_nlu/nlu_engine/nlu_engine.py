from __future__ import unicode_literals

import json
import logging
from builtins import str
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from future.utils import iteritems
from snips_nlu.tokenization import normalize

from snips_nlu.__about__ import __model_version__, __version__
from snips_nlu.constants import (
    BUILTIN_ENTITY_PARSER, CAPITALIZE, ENTITIES, GAZETTEER_ENTITIES, LANGUAGE,
    RES_ENTITY, RES_INTENT, RES_SLOTS, CUSTOM_ENTITY_PARSER, RES_VALUE,
    ENTITY_KIND, RES_MATCH_RANGE, ENTITY, VALUE, AUTOMATICALLY_EXTENSIBLE,
    UTTERANCES, DATA)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.default_configs import DEFAULT_CONFIGS
from snips_nlu.nlu_engine.utils import resolve_slots
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.pipeline.configs import NLUEngineConfig
from snips_nlu.pipeline.processing_unit import (
    MLUnit, build_ml_unit, load_processing_unit)
from snips_nlu.resources import (
    get_builtin_entity_parser_from_dir, load_resources_from_dir,
    persist_resources)
from snips_nlu.result import empty_result, is_empty, parsing_result, \
    builtin_slot, custom_slot
from snips_nlu.utils import (
    check_persisted_path, fitted_required, get_slot_name_mappings, json_string,
    log_elapsed_time, log_result)

logger = logging.getLogger(__name__)


class SnipsNLUEngine(MLUnit):
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

    unit_name = "nlu_engine"
    config_type = NLUEngineConfig

    def __init__(self, config=None, **shared):
        """The NLU engine can be configured by passing a
        :class:`.NLUEngineConfig`"""
        super(SnipsNLUEngine, self).__init__(config, **shared)
        self.intent_parsers = []
        """list of :class:`.IntentParser`"""
        self._dataset_metadata = None

    @property
    def fitted(self):
        """Whether or not the nlu engine has already been fitted"""
        return self._dataset_metadata is not None

    @log_elapsed_time(
        logger, logging.INFO, "Fitted NLU engine in {elapsed_time}")
    def fit(self, dataset, force_retrain=True):
        """Fit the NLU engine

        Args:
            dataset (dict): A valid Snips dataset
            force_retrain (bool, optional): If *False*, will not retrain intent
                parsers when they are already fitted. Default to *True*.

        Returns:
            The same object, trained.
        """
        logger.info("Fitting NLU engine...")

        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        dataset = validate_and_format_dataset(dataset)
        self._dataset_metadata = _get_dataset_metadata(dataset)

        if self.config is None:
            language = self._dataset_metadata["language_code"]
            self.config = self.config_type.from_dict(DEFAULT_CONFIGS[language])

        parsers = []
        for parser_config in self.config.intent_parsers_configs:
            # Re-use existing parsers to allow pre-training
            recycled_parser = None
            for parser in self.intent_parsers:
                if parser.unit_name == parser_config.unit_name:
                    recycled_parser = parser
                    break
            if recycled_parser is None:
                recycled_parser = build_ml_unit(parser_config)

            recycled_parser.builtin_entity_parser = self.builtin_entity_parser
            recycled_parser.custom_entity_parser = self.custom_entity_parser
            if force_retrain or not recycled_parser.fitted:
                recycled_parser.fit(dataset, force_retrain)
            parsers.append(recycled_parser)

        self.intent_parsers = parsers
        return self

    @log_result(logger, logging.DEBUG, "Result -> {result}")
    @log_elapsed_time(logger, logging.DEBUG, "Parsed query in {elapsed_time}")
    @fitted_required
    def parse(self, text, intents=None):
        """Performs intent parsing on the provided *text* by calling its intent
        parsers successively

        Args:
            text (str): Input
            intents (str or list of str): If provided, reduces the scope of
                intent parsing to the provided list of intents

        Returns:
            dict: The most likely intent along with the extracted slots. See
            :func:`.parsing_result` for the output format.

        Raises:
            NotTrained: When the nlu engine is not fitted
            TypeError: When input type is not unicode
        """
        logging.info("NLU engine parsing: '%s'...", text)
        if not isinstance(text, str):
            raise TypeError("Expected unicode but received: %s" % type(text))

        if isinstance(intents, str):
            intents = [intents]

        entities = self._dataset_metadata["entities"]

        for parser in self.intent_parsers:
            res = parser.parse(text, intents)
            if is_empty(res):
                continue
            slots = res[RES_SLOTS]
            scope = [s[RES_ENTITY] for s in slots
                     if is_builtin_entity(s[RES_ENTITY])]
            resolved_slots = resolve_slots(
                text, slots, entities, self.builtin_entity_parser, scope)
            return parsing_result(text, intent=res[RES_INTENT],
                                  slots=resolved_slots)
        return empty_result(text)

    def resolve_slots(self, text, slots, scope):
        builtin_scope = [entity_kind for entity_kind in scope
                         if is_builtin_entity(entity_kind)]
        custom_scope = [entity_kind for entity_kind in scope
                        if not is_builtin_entity(entity_kind)]
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
            if is_builtin_entity(entity_name):
                entities = builtin_entities
                parser = self.builtin_entity_parser
                slot_builder = builtin_slot
                use_cache = False
            else:
                entities = custom_entities
                parser = self.custom_entity_parser
                slot_builder = custom_slot
                use_cache = True

            resolved_slot = None
            for ent in entities:
                if ent[ENTITY_KIND] == entity_name and \
                        ent[RES_MATCH_RANGE] == slot[RES_MATCH_RANGE]:
                    resolved_slot = slot_builder(slot, ent[ENTITY])
                    break
            if resolved_slot is None:
                matches = parser.parse(
                    raw_value, scope=[entity_name], use_cache=use_cache)
                if matches:
                    resolved_slot = slot_builder(slot, matches[0][VALUE])

            extensible = self._dataset_metadata[ENTITIES][
                entity_name][AUTOMATICALLY_EXTENSIBLE]
            if not is_builtin_entity(entity_name) and resolved_slot is None \
                    and extensible:
                resolved_slot = custom_slot(slot)
            if resolved_slot is not None:
                resolved_slots.append(resolved_slot)

        return resolved_slots

    @check_persisted_path
    def persist(self, path):
        """Persist the NLU engine at the given directory path

        Args:
            path (str): the location at which the nlu engine must be persisted.
                This path must not exist when calling this function.
        """
        directory_path = Path(path)
        directory_path.mkdir()

        parsers_count = defaultdict(int)
        intent_parsers = []
        for parser in self.intent_parsers:
            parser_name = parser.unit_name
            parsers_count[parser_name] += 1
            count = parsers_count[parser_name]
            if count > 1:
                parser_name = "{n}_{c}".format(n=parser_name, c=count)
            parser_path = directory_path / parser_name
            parser.persist(parser_path)
            intent_parsers.append(parser_name)

        config = None
        if self.config is not None:
            config = self.config.to_dict()

        custom_entities_parser_path = None
        if self.custom_entity_parser is not None:
            custom_entities_parser_path = \
                directory_path / "custom_intent_parser"
            self.custom_entity_parser.persist(custom_entities_parser_path)


        model = {
            "unit_name": self.unit_name,
            "dataset_metadata": self._dataset_metadata,
            "intent_parsers": intent_parsers,
            CUSTOM_ENTITY_PARSER: custom_entities_parser_path,
            "config": config,
            "model_version": __model_version__,
            "training_package_version": __version__
        }
        model_json = json_string(model)
        model_path = directory_path / "nlu_engine.json"
        with model_path.open(mode="w") as f:
            f.write(model_json)

        if self.fitted:
            required_resources = self._get_required_resources()
            if required_resources:
                language = self._dataset_metadata["language_code"]
                resources_path = directory_path / "resources"
                resources_path.mkdir()
                persist_resources(resources_path / language,
                                  required_resources, language)

    @classmethod
    def from_path(cls, path, **shared):
        """Load a :class:`SnipsNLUEngine` instance from a directory path

        The data at the given path must have been generated using
        :func:`~SnipsNLUEngine.persist`

        Args:
            path (str): The path where the nlu engine is stored.
        """
        directory_path = Path(path)
        model_path = directory_path / "nlu_engine.json"
        if not model_path.exists():
            raise OSError("Missing nlu engine model file: %s"
                          % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model = json.load(f)
        model_version = model.get("model_version")
        if model_version is None or model_version != __model_version__:
            raise ValueError(
                "Incompatible data model: persisted object=%s, python lib=%s"
                % (model_version, __model_version__))

        resources_dir = directory_path / "resources"
        builtin_entity_parser_path = shared.get(BUILTIN_ENTITY_PARSER)
        builtin_entity_parser = None
        if builtin_entity_parser_path is None:
            if resources_dir.is_dir():
                for subdir in resources_dir.iterdir():
                    if subdir.is_dir() and (subdir / "metadata.json").exists():
                        load_resources_from_dir(subdir)
                        builtin_entity_parser = \
                            get_builtin_entity_parser_from_dir(subdir)
                        shared[BUILTIN_ENTITY_PARSER] = builtin_entity_parser
                        break

        custom_entity_parser = None
        custom_entity_parser_path = shared.get(CUSTOM_ENTITY_PARSER)
        if custom_entity_parser_path is  not None:
            custom_entity_parser = load_processing_unit(
                custom_entity_parser_path, **shared)
            shared[CUSTOM_ENTITY_PARSER] = custom_entity_parser

        nlu_engine = cls(config=model["config"],
                         builtin_entity_parser=builtin_entity_parser,
                         custom_entity_parser=custom_entity_parser)

        # pylint:disable=protected-access
        nlu_engine._dataset_metadata = model["dataset_metadata"]
        # pylint:enable=protected-access
        intent_parsers = []
        for intent_parser_name in model["intent_parsers"]:
            intent_parser_path = directory_path / intent_parser_name
            intent_parser = load_processing_unit(intent_parser_path, **shared)
            intent_parsers.append(intent_parser)
        nlu_engine.intent_parsers = intent_parsers
        return nlu_engine

    def _get_required_resources(self):
        required_resources = self.config.get_required_resources()
        if self.builtin_entity_parser is not None:
            gazetteer_entities = [
                conf["builtin_entity_name"] for conf in
                self.builtin_entity_parser.gazetteer_entity_configurations]
            required_resources[GAZETTEER_ENTITIES] = gazetteer_entities
        return required_resources


def _get_dataset_metadata(dataset):
    entities = dict()
    for entity_name, entity in iteritems(dataset[ENTITIES]):
        if is_builtin_entity(entity_name):
            continue
        ent = deepcopy(entity)
        ent.pop(CAPITALIZE)
        ent.pop(UTTERANCES)
        entities[entity_name] = ent
    slot_name_mappings = get_slot_name_mappings(dataset)
    return {
        "language_code": dataset[LANGUAGE],
        "entities": entities,
        "slot_name_mappings": slot_name_mappings
    }
