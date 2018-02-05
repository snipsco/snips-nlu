from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class SlotFiller(with_metaclass(ABCMeta, ProcessingUnit)):
    @abstractmethod
    def get_slots(self, text):
        raise NotImplementedError

    @abstractmethod
    def fit(self, dataset, intent):
        raise NotImplementedError
