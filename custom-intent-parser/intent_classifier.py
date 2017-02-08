from abc import ABCMeta, abstractmethod


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, queries):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass
