from abc import ABCMeta, abstractmethod
from snips_nlu.result import IntentClassificationResult


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass


class SnipsIntentClassifier(IntentClassifier):
    def __init__(self):
        self.intent_name = None

    def fit(self, dataset):
        intents = dataset["intents"]
        if len(intents.keys()) > 0:
            self.intent_name = intents.keys()[0]

    def get_intent(self, text):
        if self.intent_name is not None:
            return IntentClassificationResult(intent_name=self.intent_name,
                                              probability=1.0)
        return None
