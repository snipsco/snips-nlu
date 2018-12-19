from abc import abstractmethod, ABCMeta

from future.utils import with_metaclass

from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.common.abc_utils import classproperty


class IntentClassifier(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction which performs intent classification

    A custom intent classifier must inherit this class to be used in a
    :class:`.ProbabilisticIntentParser`
    """

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        return IntentClassifier.registered_name(cls)

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

    @abstractmethod
    def get_intents(self, text):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent

        .. note::

            The probabilities returned along with each intent are not
            guaranteed to sum to 1.0. They should be considered as scores
            between 0 and 1.
        """
        pass
