from abc import ABCMeta, abstractmethod, abstractproperty


class IntentClassifier(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def fitted(self):
        pass

    def check_fitted(self):
        if not self.fitted:
            raise ValueError("IntentClassifier must be fitted before "
                             "calling the 'fit' method.")

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass
