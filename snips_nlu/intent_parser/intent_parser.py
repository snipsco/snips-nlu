from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentParser(with_metaclass(ABCMeta, ProcessingUnit)):

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def parse(self, text, intents):
        pass
