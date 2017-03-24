from intent_parser import IntentParser
from snips_nlu.result import ParsedSlot
from snips_nlu.slot_filler.crf_utils import (tags_to_slots,
                                             utterance_to_sample)
from snips_nlu.tokenization import tokenize


def get_slot_name_to_entity(dataset):
    slot_name_to_entity = dict()
    for intent in dataset["intents"].values():
        for utterance in intent["utterances"]:
            for chunk in utterance["data"]:
                if "slot_name" in chunk:
                    slot_name_to_entity[chunk["slot_name"]] = chunk["entity"]
    return slot_name_to_entity


class CRFIntentParser(IntentParser):
    def __init__(self, intent_classifier, crf_taggers):
        super(CRFIntentParser, self).__init__()
        self.intent_classifier = intent_classifier
        self.crf_taggers = crf_taggers
        self.slot_name_to_entity = None

    def get_intent(self, text):
        if not self.fitted:
            raise ValueError("CRFIntentParser must be fitted before "
                             "`get_intent` is called")
        return self.intent_classifier.get_intent(text)

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        if not self.fitted:
            raise ValueError("CRFIntentParser must be fitted before "
                             "`get_slots` is called")
        if intent not in self.intent_classifier:
            raise KeyError("Invalid intent '%s'" % intent)
        tokens = tokenize(text)
        tagger = self.crf_taggers[intent]

        tags = tagger.get_tags(tokens)
        slots = tags_to_slots(tokens, tags, tagging=tagger.tagging)
        return [ParsedSlot(match_range=s["range"],
                           value=text[s["range"][0]:s["range"][1]],
                           entity=self.slot_name_to_entity[s["slot_name"]],
                           slot_name=s["slot_name"]) for s in slots]

    @property
    def fitted(self):
        return self.intent_classifier.fitted and all(
            slot_filler.fitted for slot_filler in self.crf_taggers)

    def fit(self, dataset):
        self.slot_name_to_entity = get_slot_name_to_entity(dataset)
        self.intent_classifier = self.intent_classifier.fit(dataset)

        for intent_name in dataset["intents"]:
            intent_utterances = dataset["intents"][intent_name]["utterances"]
            tagging = self.crf_taggers[intent_name].tagging
            crf_samples = [utterance_to_sample(u["data"], tagging)
                           for u in intent_utterances]
            self.crf_taggers[intent_name] = self.crf_taggers[intent_name].fit(
                crf_samples)
        return self


