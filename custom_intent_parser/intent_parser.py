from abc import ABCMeta, abstractmethod, abstractproperty


class IntentParser(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def intents(self):
        pass

    @abstractmethod
    @intents.setter
    def intents(self, value):
        pass

    @abstractproperty
    def entities(self):
        pass

    @abstractmethod
    @entities.setter
    def entities(self, value):
        pass

    @abstractproperty
    def fitted(self):
        pass

    def check_fitted(self):
        if not self.fitted:
            raise ValueError("IntentParser must be fitted before calling the"
                             " 'fit' method.")

    @abstractmethod
    def parse(self, text):
        pass

    @abstractmethod
    def get_intent(self, text):
        pass

    @abstractmethod
    def get_entities(self, text, intent=None):
        pass

    @classmethod
    @abstractmethod
    def from_dataset(cls, dataset):
        pass
