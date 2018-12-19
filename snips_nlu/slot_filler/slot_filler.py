from abc import abstractmethod, ABCMeta

from future.utils import with_metaclass

from snips_nlu.common.abc_utils import classproperty
from snips_nlu.pipeline.processing_unit import ProcessingUnit


class SlotFiller(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction which performs slot filling

    A custom slot filler must inherit this class to be used in a
    :class:`.ProbabilisticIntentParser`
    """

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        return SlotFiller.registered_name(cls)

    @abstractmethod
    def fit(self, dataset, intent):
        """Fit the slot filler with a valid Snips dataset"""
        pass

    @abstractmethod
    def get_slots(self, text):
        """Performs slot extraction (slot filling) on the provided *text*

        Returns:
            list of dict: The list of extracted slots. See
            :func:`.unresolved_slot` for the output format of a slot
        """
        pass
