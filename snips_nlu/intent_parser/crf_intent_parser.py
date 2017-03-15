from intent_parser import IntentParser
from ..intent_classifier.intent_classifier import IntentClassifier
from ..crf.crf_token_classifier import CRFTokenClassifier
from ..tokenizer import Tokenizer


class CRFIntentParser(IntentParser):
    def __init__(self, tokenizer, intent_classifier, slot_fillers):
        super(CRFIntentParser, self).__init__()
        self._tokenizer = None
        self.tokenizer = tokenizer

        self._intent_classifier = None
        self.intent_classifier = intent_classifier

        self._slot_fillers = None
        self.slot_fillers = slot_fillers

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        if not isinstance(value, Tokenizer):
            raise ValueError("Expected Tokenizer but found: %s" % type(value))
        self._tokenizer = value

    @property
    def intent_classifier(self):
        return self._intent_classifier

    @intent_classifier.setter
    def intent_classifier(self, value):
        if not isinstance(value, IntentClassifier):
            raise ValueError("Expected IntentClassifier but found: %s"
                             % type(value))
        self._intent_classifier = value

    @property
    def slot_fillers(self):
        return self._slot_fillers

    @slot_fillers.setter
    def slot_fillers(self, value):
        if not isinstance(value, dict):
            raise ValueError("Expected dict but found: %s" % type(value))
        for slot_filler in value.itervalues():
            if not isinstance(slot_filler, CRFTokenClassifier):
                raise ValueError("Expected CRFTokenClassifier but found: %s"
                                 % type(slot_filler))
        self._slot_fillers = value

    def fit(self, dataset):
        pass

    def parse(self, text):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
