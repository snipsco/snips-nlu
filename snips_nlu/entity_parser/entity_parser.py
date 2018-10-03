# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from copy import deepcopy

from future.builtins import object
from future.utils import with_metaclass

from snips_nlu.utils import LimitedSizeDict

# pylint: disable=ungrouped-imports

try:
    from abc import abstractclassmethod
except ImportError:
    from snips_nlu.utils import abstractclassmethod


# pylint: enable=ungrouped-imports


class EntityParser(with_metaclass(ABCMeta, object)):
    def __init__(self, parser):
        self._parser = parser
        self._cache = LimitedSizeDict(size_limit=1000)

    def parse(self, text, scope=None, use_cache=True):
        text = text.lower()
        if not use_cache:
            return self._parser.parse(text, scope)
        scope_key = tuple(sorted(scope)) if scope is not None else scope
        cache_key = (text, scope_key)
        if cache_key not in self._cache:
            parser_result = self._parser.parse(text, scope)
            self._cache[cache_key] = parser_result
        return deepcopy(self._cache[cache_key])

    @abstractmethod
    def persist(self, path):
        pass

    @abstractclassmethod
    def from_path(cls, path):
        pass
