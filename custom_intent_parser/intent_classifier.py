from abc import ABCMeta, abstractmethod, abstractproperty


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def intents(self):
        pass

    @abstractproperty
    def fitted(self):
        pass

    def check_fitted(self):
        if not self.fitted:
            raise ValueError("IntentClassifier must be fitted before "
                             "calling the 'fit' method.")

    @abstractmethod
    def fit(self, queries):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass

    @classmethod
    @abstractmethod
    def from_dataset(cls, dataset):
        pass
