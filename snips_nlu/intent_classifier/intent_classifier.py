from abc import ABCMeta, abstractmethod

from snips_nlu.pipeline.processing_unit import ProcessingUnit
from future.utils import with_metaclass


class IntentClassifier(with_metaclass(ABCMeta, ProcessingUnit)):
    @abstractmethod
    def get_intent(self, text, intents_filter):
        pass

    @abstractmethod
    def fit(self, dataset):
        pass
