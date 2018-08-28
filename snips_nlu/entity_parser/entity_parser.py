# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractproperty

from future.utils import with_metaclass
from future.builtins import object

from snips_nlu.utils import LimitedSizeDict


class EntityParser(with_metaclass(ABCMeta, object)):
    @abstractproperty
    def parser(self):
        pass

    @property
    def cache(self):
        try:
            return self._cache
        except AttributeError:
            self._cache = LimitedSizeDict(size_limit=1000)
        return self._cache

    def parse(self, text, scope=None, use_cache=True):
        text = text.lower()
        if not use_cache:
            return self.parser.parse(text, scope)
        cache_key = (text, tuple(sorted(scope)))
        if cache_key not in self.cache:
            parser_result = self.parser.parse(text, scope)
            self.cache[cache_key] = parser_result
        return self.cache[cache_key]
