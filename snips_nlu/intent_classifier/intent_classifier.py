from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentClassifier(with_metaclass(ABCMeta, ProcessingUnit)):
    @abstractmethod
    def get_intent(self, text, intents_filter):
        pass

    @abstractmethod
    def fit(self, dataset):
        pass
