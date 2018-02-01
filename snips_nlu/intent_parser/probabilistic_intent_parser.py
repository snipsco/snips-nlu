from __future__ import unicode_literals

from copy import deepcopy

from future.utils import itervalues, iteritems

from snips_nlu.constants import INTENTS, RES_INTENT_NAME
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.intent_parser.intent_parser import IntentParser
from snips_nlu.pipeline.configs.intent_parser import \
    ProbabilisticIntentParserConfig
from snips_nlu.pipeline.processing_unit import (
    build_processing_unit, load_processing_unit)
from snips_nlu.result import empty_result, parsing_result
from snips_nlu.utils import NotTrained


class ProbabilisticIntentParser(IntentParser):
    unit_name = "probabilistic_intent_parser"
    config_type = ProbabilisticIntentParserConfig

    def __init__(self, config=None):
        if config is None:
            config = self.config_type()
        super(ProbabilisticIntentParser, self).__init__(config)
        self.intent_classifier = None
        self.slot_fillers = dict()


    @property
    def fitted(self):
        return self.intent_classifier is not None \
               and self.intent_classifier.fitted \
               and all(slot_filler is not None and slot_filler.fitted
                       for slot_filler in itervalues(self.slot_fillers))

    def fit(self, dataset, force_retrain=True):
        dataset = validate_and_format_dataset(dataset)
        intents = list(dataset[INTENTS])
        if self.intent_classifier is None:
            self.intent_classifier = build_processing_unit(
                self.config.intent_classifier_config)
        if force_retrain or not self.intent_classifier.fitted:
            self.intent_classifier.fit(dataset)

        if self.slot_fillers is None:
            self.slot_fillers = dict()
        for intent_name in intents:
            # We need to copy the slot filler config as it may be mutated
            if self.slot_fillers.get(intent_name) is None:
                slot_filler_config = deepcopy(self.config.slot_filler_config)
                self.slot_fillers[intent_name] = build_processing_unit(
                    slot_filler_config)
            if force_retrain or not self.slot_fillers[intent_name].fitted:
                self.slot_fillers[intent_name].fit(dataset, intent_name)
        return self

    def parse(self, text, intents=None):
        if not self.fitted:
            raise NotTrained("ProbabilisticIntentParser must be fitted")

        if isinstance(intents, str):
            intents = [intents]

        intent_result = self.intent_classifier.get_intent(text, intents)
        if intent_result is None:
            return empty_result(text)

        intent_name = intent_result[RES_INTENT_NAME]
        slots = self.slot_fillers[intent_name].get_slots(text)
        return parsing_result(text, intent_result, slots)

    def to_dict(self):
        intent_classifier_dict = None

        if self.intent_classifier is not None:
            intent_classifier_dict = self.intent_classifier.to_dict()

        slot_fillers = {
            intent: slot_filler.to_dict()
            for intent, slot_filler in iteritems(self.slot_fillers)}

        return {
            "unit_name": self.unit_name,
            "intent_classifier": intent_classifier_dict,
            "config": self.config.to_dict(),
            "slot_fillers": slot_fillers,
        }

    @classmethod
    def from_dict(cls, unit_dict):
        slot_fillers = {
            intent: load_processing_unit(slot_filler_dict) for
            intent, slot_filler_dict in
            iteritems(unit_dict["slot_fillers"])}
        classifier = None
        if unit_dict["intent_classifier"] is not None:
            classifier = load_processing_unit(unit_dict["intent_classifier"])

        parser = cls(config=cls.config_type.from_dict(unit_dict["config"]))
        parser.intent_classifier = classifier
        parser.slot_fillers = slot_fillers
        return parser
