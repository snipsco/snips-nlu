from abc import ABCMeta, abstractmethod


class SlotFiller(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataset, intent_name):
        pass

    @abstractmethod
    def get_slots(self, text):
        pass
