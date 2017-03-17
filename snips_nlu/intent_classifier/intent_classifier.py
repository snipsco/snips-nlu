from abc import ABCMeta, abstractmethod


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    def __init__(self, intent_name):
        self.intent_name = intent_name

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass
