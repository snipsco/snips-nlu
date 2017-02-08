from abc import ABCMeta, abstractmethod


class SlotFiller(object):
    __metaclass__ = ABCMeta

    _slots = []

    @property
    def slots(self):
        return self._slots

    @slots.setter
    def slots(self, value):
        self._slots = value

    @abstractmethod
    def fit(self, queries):
        pass

    @abstractmethod
    def get_slots(self, text):
        pass
