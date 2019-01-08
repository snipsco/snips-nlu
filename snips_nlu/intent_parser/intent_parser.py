from abc import abstractmethod, ABCMeta

from future.utils import with_metaclass

from snips_nlu.common.abc_utils import classproperty
from snips_nlu.pipeline.processing_unit import ProcessingUnit


class IntentParser(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction which performs intent parsing

    A custom intent parser must inherit this class to be used in a
    :class:`.SnipsNLUEngine`
    """

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        return IntentParser.registered_name(cls)

    @abstractmethod
    def fit(self, dataset, force_retrain):
        """Fit the intent parser with a valid Snips dataset

        Args:
            dataset (dict): valid Snips NLU dataset
            force_retrain (bool): specify whether or not sub units of the
            intent parser that may be already trained should be retrained
        """
        pass

    @abstractmethod
    def parse(self, text, intents, top_n):
        """Performs intent parsing on the provided *text*

        Args:
            text (str): input
            intents (str or list of str): if provided, reduces the scope of
                intent parsing to the provided list of intents
            top_n (int, optional): when provided, this method will return a
                list of at most top_n most likely intents, instead of a single
                parsing result.
                Note that the returned list can contain less than ``top_n``
                elements, for instance when the parameter ``intents`` is not
                None, or when ``top_n`` is greater than the total number of
                intents.

        Returns:
            dict or list: the most likely intent(s) along with the extracted
            slots. See :func:`.parsing_result` and :func:`.extraction_result`
            for the output format.
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

    @abstractmethod
    def get_slots(self, text, intent):
        """Extract slots from a text input, with the knowledge of the intent

        Args:
            text (str): input
            intent (str): the intent which the input corresponds to

        Returns:
            list: the list of extracted slots

        Raises:
            IntentNotFoundError: when the intent was not part of the training
                data
        """
        pass
