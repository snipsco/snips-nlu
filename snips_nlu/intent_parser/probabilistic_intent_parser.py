from __future__ import unicode_literals

import json
import logging
from builtins import str
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from future.utils import iteritems, itervalues

from snips_nlu.constants import INTENTS, RES_INTENT_NAME
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.pipeline.configs import ProbabilisticIntentParserConfig
from snips_nlu.pipeline.processing_unit import (
    load_processing_unit, build_processing_unit)
from snips_nlu.result import empty_result, parsing_result
from snips_nlu.utils import (check_persisted_path, elapsed_since,
                             fitted_required, json_string, log_elapsed_time,
                             log_result)

logger = logging.getLogger(__name__)


class ProbabilisticIntentParser(IntentParser):
    """Intent parser which consists in two steps: intent classification then
    slot filling"""

    unit_name = "probabilistic_intent_parser"
    config_type = ProbabilisticIntentParserConfig

    # pylint:disable=line-too-long
    def __init__(self, config=None, **shared):
        """The probabilistic intent parser can be configured by passing a
        :class:`.ProbabilisticIntentParserConfig`"""
        if config is None:
            config = self.config_type()
        super(ProbabilisticIntentParser, self).__init__(config, **shared)
        self.intent_classifier = None
        self.slot_fillers = dict()

    # pylint:enable=line-too-long

    @property
    def fitted(self):
        """Whether or not the intent parser has already been fitted"""
        return self.intent_classifier is not None \
               and self.intent_classifier.fitted \
               and all(slot_filler is not None and slot_filler.fitted
                       for slot_filler in itervalues(self.slot_fillers))

    @log_elapsed_time(logger, logging.INFO,
                      "Fitted probabilistic intent parser in {elapsed_time}")
    # pylint:disable=arguments-differ
    def fit(self, dataset, force_retrain=True):
        """Fit the slot filler

        Args:
            dataset (dict): A valid Snips dataset
            force_retrain (bool, optional): If *False*, will not retrain intent
                classifier and slot fillers when they are already fitted.
                Default to *True*.

        Returns:
            :class:`ProbabilisticIntentParser`: The same instance, trained
        """
        logger.info("Fitting probabilistic intent parser...")
        dataset = validate_and_format_dataset(dataset)
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        intents = list(dataset[INTENTS])
        if self.intent_classifier is None:
            self.intent_classifier = build_processing_unit(
                self.config.intent_classifier_config)
        self.intent_classifier.builtin_entity_parser = \
            self.builtin_entity_parser
        if force_retrain or not self.intent_classifier.fitted:
            self.intent_classifier.fit(dataset)

        if self.slot_fillers is None:
            self.slot_fillers = dict()
        slot_fillers_start = datetime.now()
        for intent_name in intents:
            # We need to copy the slot filler config as it may be mutated
            if self.slot_fillers.get(intent_name) is None:
                slot_filler_config = deepcopy(self.config.slot_filler_config)
                self.slot_fillers[intent_name] = build_processing_unit(
                    slot_filler_config)
            self.slot_fillers[intent_name].builtin_entity_parser = \
                self.builtin_entity_parser
            if force_retrain or not self.slot_fillers[intent_name].fitted:
                self.slot_fillers[intent_name].fit(dataset, intent_name)
        logger.debug("Fitted slot fillers in %s",
                     elapsed_since(slot_fillers_start))
        return self

    # pylint:enable=arguments-differ

    @log_result(logger, logging.DEBUG,
                "ProbabilisticIntentParser result -> {result}")
    @log_elapsed_time(logger, logging.DEBUG,
                      "ProbabilisticIntentParser parsed in {elapsed_time}")
    @fitted_required
    def parse(self, text, intents=None):
        """Performs intent parsing on the provided *text* by first classifying
        the intent and then using the correspond slot filler to extract slots

        Args:
            text (str): Input
            intents (str or list of str): If provided, reduces the scope of
                intent parsing to the provided list of intents

        Returns:
            dict: The most likely intent along with the extracted slots. See
            :func:`.parsing_result` for the output format.

        Raises:
            NotTrained: When the intent parser is not fitted
        """
        logger.debug("Probabilistic intent parser parsing '%s'...", text)

        if isinstance(intents, str):
            intents = [intents]

        intent_result = self.intent_classifier.get_intent(text, intents)
        if intent_result is None:
            return empty_result(text)

        intent_name = intent_result[RES_INTENT_NAME]
        slots = self.slot_fillers[intent_name].get_slots(text)
        return parsing_result(text, intent_result, slots)

    @check_persisted_path
    def persist(self, path):
        """Persist the object at the given path"""
        path = Path(path)
        path.mkdir()
        sorted_slot_fillers = sorted(iteritems(self.slot_fillers))
        slot_fillers = []
        for i, (intent, slot_filler) in enumerate(sorted_slot_fillers):
            slot_filler_name = "slot_filler_%s" % i
            slot_filler.persist(path / slot_filler_name)
            slot_fillers.append({
                "intent": intent,
                "slot_filler_name": slot_filler_name
            })

        if self.intent_classifier is not None:
            self.intent_classifier.persist(path / "intent_classifier")

        model = {
            "config": self.config.to_dict(),
            "slot_fillers": slot_fillers
        }
        model_json = json_string(model)
        model_path = path / "intent_parser.json"
        with model_path.open(mode="w") as f:
            f.write(model_json)
        self.persist_metadata(path)

    @classmethod
    def from_path(cls, path, **shared):
        """Load a :class:`ProbabilisticIntentParser` instance from a path

        The data at the given path must have been generated using
        :func:`~ProbabilisticIntentParser.persist`
        """
        path = Path(path)
        model_path = path / "intent_parser.json"
        if not model_path.exists():
            raise OSError("Missing probabilistic intent parser model file: "
                          "%s" % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model = json.load(f)

        parser = cls(config=cls.config_type.from_dict(model["config"]),
                     **shared)
        classifier = None
        intent_classifier_path = path / "intent_classifier"
        if intent_classifier_path.exists():
            classifier = load_processing_unit(intent_classifier_path, **shared)

        slot_fillers = dict()
        for slot_filler_conf in model["slot_fillers"]:
            intent = slot_filler_conf["intent"]
            slot_filler_path = path / slot_filler_conf["slot_filler_name"]
            slot_filler = load_processing_unit(slot_filler_path, **shared)
            slot_fillers[intent] = slot_filler

        parser.intent_classifier = classifier
        parser.slot_fillers = slot_fillers
        return parser
