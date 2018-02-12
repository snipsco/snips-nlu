from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass

from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentClassifier(with_metaclass(ABCMeta, ProcessingUnit)):
    # pylint:disable=line-too-long
    """Abstraction which performs intent classification

    A custom intent classifier must inherit this class to be used in a
    :class:`.ProbabilisticIntentParser`
    """

    # pylint:enable=line-too-long

    @abstractmethod
    def fit(self, dataset):
        """Fit the intent classifier with a valid Snips dataset"""
        pass

    @abstractmethod
    def get_intent(self, text, intents_filter):
        """Performs intent classification on the provided *text*

        Args:
            text (str): Input
            intents_filter (str or list of str): When defined, it will find
                the most likely intent among the list, otherwise it will use
                the whole list of intents defined in the dataset

        Returns:
            dict or None: The most likely intent along with its probability or
            *None* if no intent was found. See
            :func:`.intent_classification_result` for the output format.
        """
        pass
