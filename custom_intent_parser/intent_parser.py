from abc import ABCMeta, abstractmethod, abstractproperty


class IntentParser(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def fitted(self):
        pass

    def check_fitted(self):
        if not self.fitted:
            raise ValueError("IntentParser must be fitted before calling the"
                             " 'fit' method.")

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def parse(self, text):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def get_entities(self, text, intent=None):
        pass
