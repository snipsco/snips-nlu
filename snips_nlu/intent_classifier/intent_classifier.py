from abc import ABCMeta, abstractmethod


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataset, intent_name):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass
