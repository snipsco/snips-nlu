from __future__ import unicode_literals

import json
import logging
from builtins import str
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from future.utils import iteritems, itervalues

from snips_nlu.common.log_utils import log_elapsed_time, log_result
from snips_nlu.common.utils import (
    check_persisted_path, elapsed_since, fitted_required, json_string)
from snips_nlu.constants import INTENTS, RES_INTENT_NAME
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.exceptions import IntentNotFoundError, LoadingError
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.pipeline.configs import ProbabilisticIntentParserConfig
from snips_nlu.result import parsing_result, extraction_result
from snips_nlu.slot_filler import SlotFiller

logger = logging.getLogger(__name__)


@IntentParser.register("probabilistic_intent_parser")
class ProbabilisticIntentParser(IntentParser):
    """Intent parser which consists in two steps: intent classification then
    slot filling"""

    config_type = ProbabilisticIntentParserConfig

    def __init__(self, config=None, **shared):
        """The probabilistic intent parser can be configured by passing a
        :class:`.ProbabilisticIntentParserConfig`"""
        super(ProbabilisticIntentParser, self).__init__(config, **shared)
        self.intent_classifier = None
        self.slot_fillers = dict()

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
        """Fits the probabilistic intent parser

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
        intents = list(dataset[INTENTS])
        if self.intent_classifier is None:
            self.intent_classifier = IntentClassifier.from_config(
                self.config.intent_classifier_config,
                builtin_entity_parser=self.builtin_entity_parser,
                custom_entity_parser=self.custom_entity_parser,
                resources=self.resources)

        if force_retrain or not self.intent_classifier.fitted:
            self.intent_classifier.fit(dataset)

        if self.slot_fillers is None:
            self.slot_fillers = dict()
        slot_fillers_start = datetime.now()
        for intent_name in intents:
            # We need to copy the slot filler config as it may be mutated
            if self.slot_fillers.get(intent_name) is None:
                slot_filler_config = deepcopy(self.config.slot_filler_config)
                self.slot_fillers[intent_name] = SlotFiller.from_config(
                    slot_filler_config,
                    builtin_entity_parser=self.builtin_entity_parser,
                    custom_entity_parser=self.custom_entity_parser,
                    resources=self.resources)
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
    def parse(self, text, intents=None, top_n=None):
        """Performs intent parsing on the provided *text* by first classifying
        the intent and then using the correspond slot filler to extract slots

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
            intents = {intents}
        elif isinstance(intents, list):
            intents = list(intents)

        if top_n is None:
            intent_result = self.intent_classifier.get_intent(text, intents)
            intent_name = intent_result[RES_INTENT_NAME]
            if intent_name is not None:
                slots = self.slot_fillers[intent_name].get_slots(text)
            else:
                slots = []
            return parsing_result(text, intent_result, slots)

        results = []
        intents_results = self.intent_classifier.get_intents(text)
        for intent_result in intents_results[:top_n]:
            intent_name = intent_result[RES_INTENT_NAME]
            if intent_name is not None:
                slots = self.slot_fillers[intent_name].get_slots(text)
            else:
                slots = []
            results.append(extraction_result(intent_result, slots))
        return results

    @fitted_required
    def get_intents(self, text):
        """Returns the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent
        """
        return self.intent_classifier.get_intents(text)

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

        if intent not in self.slot_fillers:
            raise IntentNotFoundError(intent)
        return self.slot_fillers[intent].get_slots(text)

    @check_persisted_path
    def persist(self, path):
        """Persists the object at the given path"""
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
        """Loads a :class:`ProbabilisticIntentParser` instance from a path

        The data at the given path must have been generated using
        :func:`~ProbabilisticIntentParser.persist`
        """
        path = Path(path)
        model_path = path / "intent_parser.json"
        if not model_path.exists():
            raise LoadingError(
                "Missing probabilistic intent parser model file: %s"
                % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model = json.load(f)

        config = cls.config_type.from_dict(model["config"])
        parser = cls(config=config, **shared)
        classifier = None
        intent_classifier_path = path / "intent_classifier"
        if intent_classifier_path.exists():
            classifier_unit_name = config.intent_classifier_config.unit_name
            classifier = IntentClassifier.load_from_path(
                intent_classifier_path, classifier_unit_name, **shared)

        slot_fillers = dict()
        slot_filler_unit_name = config.slot_filler_config.unit_name
        for slot_filler_conf in model["slot_fillers"]:
            intent = slot_filler_conf["intent"]
            slot_filler_path = path / slot_filler_conf["slot_filler_name"]
            slot_filler = SlotFiller.load_from_path(
                slot_filler_path, slot_filler_unit_name, **shared)
            slot_fillers[intent] = slot_filler

        parser.intent_classifier = classifier
        parser.slot_fillers = slot_fillers
        return parser
