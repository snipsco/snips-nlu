from abc import ABCMeta, abstractmethod


class IntentParser(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_intent(self, text):
        raise NotImplementedError

    @abstractmethod
    def get_slots(self, text, intent=None):
        raise NotImplementedError
