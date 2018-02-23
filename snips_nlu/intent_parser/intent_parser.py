from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentParser(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction which performs intent parsing

    A custom intent parser must inherit this class to be used in a
    :class:`.SnipsNLUEngine`
    """

    @abstractmethod
    def fit(self, dataset):
        """Fit the intent parser with a valid Snips dataset"""
        pass

    @abstractmethod
    def parse(self, text, intents):
        """Performs intent parsing on the provide *text*

        Args:
            text (str): Input
            intents (str or list of str): If provided, reduces the scope of
            intent parsing to the provided list of intents

        Returns:
            dict: The most likely intent along with the extracted slots. See
            :func:`.parsing_result` for the output format.
        """
        pass
