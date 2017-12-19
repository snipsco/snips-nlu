from abc import ABCMeta, abstractmethod

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentParser(ProcessingUnit):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataset, intents):
        raise NotImplementedError

    @abstractmethod
    def get_intent(self, text):
        raise NotImplementedError

    @abstractmethod
    def get_slots(self, text, intent):
        raise NotImplementedError
