# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractproperty

from future.utils import with_metaclass

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
        cache_key = (text, str(scope))
        if cache_key not in self.cache:
            parser_result = self.parser.parse(text, scope)
            self.cache[cache_key] = parser_result
        return self.cache[cache_key]


def _get_caching_key(language, entity_scope):
    tuple_key = (language,)
    tuple_key += tuple(entity for entity in sorted(entity_scope))
    return tuple_key
