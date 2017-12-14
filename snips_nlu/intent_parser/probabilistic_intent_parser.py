from __future__ import unicode_literals

from snips_nlu.constants import INTENTS, LANGUAGE
from snips_nlu.configs.intent_parser import ProbabilisticIntentParserConfig
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_slot_filler import CRFSlotFiller
from snips_nlu.slot_filler.feature_functions import crf_features


class ProbabilisticIntentParser(object):
    def __init__(self, config=None):
        if config is None:
            config = ProbabilisticIntentParserConfig()
        self.intent_classifier = None
        self.slot_fillers = None
        self.config = config

    def get_intent(self, text):
        if not self.fitted:
            raise ValueError("ProbabilisticIntentParser must be fitted before "
                             "`get_intent` is called")
        return self.intent_classifier.get_intent(text)

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        if not self.fitted:
            raise ValueError("ProbabilisticIntentParser must be fitted before "
                             "`get_slots` is called")
        if intent not in self.slot_fillers:
            raise KeyError("Invalid intent '%s'" % intent)

        return self.slot_fillers[intent].get_slots(text)

    @property
    def fitted(self):
        return self.intent_classifier is not None \
               and self.intent_classifier.fitted \
               and all(slot_filler is not None and slot_filler.fitted
                       for slot_filler in self.slot_fillers.values())

    def fit(self, dataset, intents=None):
        missing_intents = self.get_missing_intents(dataset, intents)
        if missing_intents:
            raise ValueError("These intents must be trained: %s"
                             % missing_intents)
        if intents is None:
            intents = dataset[INTENTS].keys()

        self.language = Language.from_iso_code(dataset[LANGUAGE])
        self.intent_classifier = SnipsIntentClassifier(
            self.config.intent_classifier_config)
        self.intent_classifier.fit(dataset)
        if self.slot_fillers is None:
            self.slot_fillers = dict()
        for intent_name in intents:
            feature_signatures = crf_features(
                dataset, intent_name, self.language,
                self.config.crf_slot_filler_config)
            self.slot_fillers[intent_name] = CRFSlotFiller(
                features_signatures=feature_signatures,
                config=self.config.crf_slot_filler_config)
            self.slot_fillers[intent_name].fit(dataset, intent_name)
        return self

    def get_missing_intents(self, dataset, intents_to_fit):
        if intents_to_fit is None:
            return set()
        all_intents = set(dataset[INTENTS].keys())
        implicit_fitted_intents = all_intents.difference(intents_to_fit)
        if self.slot_fillers is None:
            already_fitted_intents = set()
        else:
            already_fitted_intents = set(
                intent_name for intent_name, slot_filler
                in self.slot_fillers.iteritems() if slot_filler.fitted)
        missing_intents = implicit_fitted_intents.difference(
            already_fitted_intents)
        return missing_intents

    def add_fitted_slot_filler(self, intent, slot_filler_data):
        if self.slot_fillers is None:
            self.slot_fillers = dict()
        self.slot_fillers[intent] = CRFSlotFiller.from_dict(slot_filler_data)

    def get_fitted_slot_filler(self, dataset, intent):
        language = Language.from_iso_code(dataset[LANGUAGE])
        crf_signatures = crf_features(dataset, intent, language,
                                      self.config.crf_slot_filler_config)
        slot_filler = CRFSlotFiller(crf_signatures,
                                    self.config.crf_slot_filler_config)
        return slot_filler.fit(dataset, intent)

    def to_dict(self):
        slot_fillers = None
        language_code = None
        intent_classifier_dict = None

        if self.language is not None:
            language_code = self.language.iso_code
        if self.intent_classifier is not None:
            intent_classifier_dict = self.intent_classifier.to_dict()
        if self.slot_fillers is not None:
            slot_fillers = {
                intent: slot_filler.to_dict()
                for intent, slot_filler in self.slot_fillers.iteritems()}

        return {
            "language_code": language_code,
            "intent_classifier": intent_classifier_dict,
            "config": self.config.to_dict(),
            "slot_fillers": slot_fillers,
        }

    @classmethod
    def from_dict(cls, obj_dict):
        slot_fillers = None
        if obj_dict["slot_fillers"] is not None:
            slot_fillers = {
                intent: CRFSlotFiller.from_dict(slot_filler_dict) for
                intent, slot_filler_dict in
                obj_dict["slot_fillers"].iteritems()}
        language = None
        if obj_dict["language_code"] is not None:
            language = Language.from_iso_code(obj_dict["language_code"])
        classifier = None
        if obj_dict["intent_classifier"] is not None:
            classifier = SnipsIntentClassifier.from_dict(
                obj_dict["intent_classifier"])

        parser = cls(config=ProbabilisticIntentParserConfig.from_dict(
            obj_dict["config"]))
        parser.language = language
        parser.intent_classifier = classifier
        parser.slot_fillers = slot_fillers
        return parser
