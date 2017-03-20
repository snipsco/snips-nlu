from abc import ABCMeta, abstractmethod
from collections import namedtuple

Token = namedtuple('Token', 'value', 'start', 'end')


class Tokenizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def tokenize(self):
        pass
