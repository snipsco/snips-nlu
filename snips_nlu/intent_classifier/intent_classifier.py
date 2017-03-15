from abc import ABCMeta, abstractmethod


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_intent(self, text, tokens):
        pass
