from abc import ABCMeta, abstractmethod

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentParser(ProcessingUnit):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataset, intents):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def get_slots(self, text, intent):
        pass
