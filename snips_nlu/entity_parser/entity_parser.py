# coding=utf-8
from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod

from future.builtins import object
from future.utils import with_metaclass

from snips_nlu.common.dict_utils import LimitedSizeDict

# pylint: disable=ungrouped-imports

try:
    from abc import abstractclassmethod
except ImportError:
    from snips_nlu.common.abc_utils import abstractclassmethod


# pylint: enable=ungrouped-imports


class EntityParser(with_metaclass(ABCMeta, object)):
    """Abstraction of a entity parser implementing some basic caching
    """

    def __init__(self):
        self._cache = LimitedSizeDict(size_limit=1000)

    def parse(self, text, scope=None, use_cache=True):
        """Search the given text for entities defined in the scope. If no
        scope is provided, search for all kinds of entities.

            Args:
                text (str): input text
                scope (list or set of str, optional): if provided the parser
                    will only look for entities which entity kind is given in
                    the scope. By default the scope is None and the parser
                    will search for all kinds of supported entities
                use_cache (bool): if False the internal cache will not be use,
                    this can be useful if the output of the parser depends on
                    the current timestamp. Defaults to True.

            Returns:
                list of dict: list of the parsed entities formatted as a dict
                    containing the string value, the resolved value, the
                    entity kind and the entity range
        """
        if not use_cache:
            return self._parse(text, scope)
        scope_key = tuple(sorted(scope)) if scope is not None else scope
        cache_key = (text, scope_key)
        if cache_key not in self._cache:
            parser_result = self._parse(text, scope)
            self._cache[cache_key] = parser_result
        return self._cache[cache_key]

    @abstractmethod
    def _parse(self, text, scope=None):
        """Internal parse method to implement in each subclass of
         :class:`.EntityParser`

            Args:
                text (str): input text
                scope (list or set of str, optional): if provided the parser
                    will only look for entities which entity kind is given in
                    the scope. By default the scope is None and the parser
                    will search for all kinds of supported entities
                use_cache (bool): if False the internal cache will not be use,
                    this can be useful if the output of the parser depends on
                    the current timestamp. Defaults to True.

            Returns:
                list of dict: list of the parsed entities. These entity must
                    have the same output format as the
                    :func:`snips_nlu.utils.result.parsed_entity` function
        """
        pass

    @abstractmethod
    def persist(self, path):
        pass

    @abstractclassmethod
    def from_path(cls, path):
        pass
