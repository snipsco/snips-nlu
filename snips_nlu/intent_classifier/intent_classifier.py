from abc import ABCMeta, abstractmethod

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentClassifier(ProcessingUnit):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_intent(self, text):
        raise NotImplementedError

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError
