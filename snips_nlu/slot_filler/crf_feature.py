from abc import ABCMeta, abstractmethod


class CRFFeature(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def compute(self, token):
        pass
