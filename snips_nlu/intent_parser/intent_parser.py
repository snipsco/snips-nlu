from abc import ABCMeta, abstractmethod

from ..result import intent_classification_result


class IntentParser(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, text):
        pass

    def get_intent(self, text):
        parsing = self.parse(text)
        intent = parsing["intent"]
        prob = None
        if intent is not None:
            prob = intent["prob"]
            intent = intent["name"]
        return intent_classification_result(intent, prob)

    def get_entities(self, text, intent=None):
        parsing = self.parse(text)
        return parsing["entities"]
