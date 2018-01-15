from abc import ABCMeta, abstractmethod

from snips_nlu.pipeline.processing_unit import ProcessingUnit
from future.utils import with_metaclass


class IntentParser(with_metaclass(ABCMeta, ProcessingUnit)):
    @abstractmethod
    def fit(self, dataset, intents):
        pass

    @abstractmethod
    def get_intent(self, text, intents):
        pass

    @abstractmethod
    def get_slots(self, text, intent):
        pass
