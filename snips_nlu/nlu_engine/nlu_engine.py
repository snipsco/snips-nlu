from __future__ import unicode_literals

import logging
from builtins import str
from copy import deepcopy

from future.utils import iteritems

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (
    ENTITIES, CAPITALIZE, LANGUAGE, RES_SLOTS, RES_ENTITY, RES_INTENT)
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.default_configs import DEFAULT_CONFIGS
from snips_nlu.nlu_engine.utils import resolve_slots
from snips_nlu.pipeline.configs import NLUEngineConfig
from snips_nlu.pipeline.processing_unit import (
    ProcessingUnit, build_processing_unit, load_processing_unit)
from snips_nlu.resources import load_resources
from snips_nlu.result import empty_result, is_empty, parsing_result
from snips_nlu.utils import (
    get_slot_name_mappings, NotTrained, log_result, log_elapsed_time)
from snips_nlu.version import __model_version__, __version__

logger = logging.getLogger(__name__)


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

    unit_name = "nlu_engine"
    config_type = NLUEngineConfig

    def __init__(self, config=None):
        """The NLU engine can be configured by passing a
        :class:`.NLUEngineConfig`"""
        super(SnipsNLUEngine, self).__init__(config)
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
                recycled_parser = build_processing_unit(parser_config)
            if force_retrain or not recycled_parser.fitted:
                recycled_parser.fit(dataset, force_retrain)
            parsers.append(recycled_parser)

        self.intent_parsers = parsers
        return self

    @log_result(logger, logging.DEBUG, "Result -> {result}")
    @log_elapsed_time(logger, logging.DEBUG, "Parsed query in {elapsed_time}")
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

        if not self.fitted:
            raise NotTrained("SnipsNLUEngine must be fitted")

        if isinstance(intents, str):
            intents = [intents]

        language = self._dataset_metadata["language_code"]
        entities = self._dataset_metadata["entities"]

        for parser in self.intent_parsers:
            res = parser.parse(text, intents)
            if is_empty(res):
                continue
            slots = res[RES_SLOTS]
            scope = [s[RES_ENTITY] for s in slots
                     if is_builtin_entity(s[RES_ENTITY])]
            resolved_slots = resolve_slots(text, slots, entities, language,
                                           scope)
            return parsing_result(text, intent=res[RES_INTENT],
                                  slots=resolved_slots)
        return empty_result(text)

    def to_dict(self):
        """Returns a json-serializable dict"""
        intent_parsers = [parser.to_dict() for parser in self.intent_parsers]
        config = None
        if self.config is not None:
            config = self.config.to_dict()
        return {
            "unit_name": self.unit_name,
            "dataset_metadata": self._dataset_metadata,
            "intent_parsers": intent_parsers,
            "config": config,
            "model_version": __model_version__,
            "training_package_version": __version__
        }

    @classmethod
    def from_dict(cls, unit_dict):
        """Creates a :class:`SnipsNLUEngine` instance from a dict

        The dict must have been generated with :func:`~SnipsNLUEngine.to_dict`

        Raises:
            ValueError: When there is a mismatch with the model version
        """
        model_version = unit_dict.get("model_version")
        if model_version is None or model_version != __model_version__:
            raise ValueError(
                "Incompatible data model: persisted object=%s, python lib=%s"
                % (model_version, __model_version__))
        dataset_metadata = unit_dict["dataset_metadata"]
        if dataset_metadata is not None:
            load_resources(dataset_metadata["language_code"])
        nlu_engine = cls(config=unit_dict["config"])
        # pylint:disable=protected-access
        nlu_engine._dataset_metadata = dataset_metadata
        # pylint:enable=protected-access
        nlu_engine.intent_parsers = [
            load_processing_unit(parser_dict)
            for parser_dict in unit_dict["intent_parsers"]]

        return nlu_engine


def _get_dataset_metadata(dataset):
    entities = dict()
    for entity_name, entity in iteritems(dataset[ENTITIES]):
        if is_builtin_entity(entity_name):
            continue
        ent = deepcopy(entity)
        ent.pop(CAPITALIZE)
        entities[entity_name] = ent
    slot_name_mappings = get_slot_name_mappings(dataset)
    return {
        "language_code": dataset[LANGUAGE],
        "entities": entities,
        "slot_name_mappings": slot_name_mappings
    }
