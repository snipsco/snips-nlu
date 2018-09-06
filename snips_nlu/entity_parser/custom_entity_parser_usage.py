from __future__ import unicode_literals

from enum import Enum, unique


@unique
class CustomEntityParserUsage(Enum):
    WITH_STEMS = 0
    """The parser is used with stemming"""
    WITHOUT_STEMS = 1
    """The parser is used without stemming"""
    WITH_AND_WITHOUT_STEMS = 2
    """The parser is used both with and without stemming"""

    @classmethod
    def merge_usages(cls, lhs_usage, rhs_usage):
        if lhs_usage is None:
            return rhs_usage
        if rhs_usage is None:
            return lhs_usage
        if lhs_usage == rhs_usage:
            return lhs_usage
        return cls.WITH_AND_WITHOUT_STEMS
