from __future__ import unicode_literals

from future.builtins import object

from snips_nlu.builtin_entities import is_builtin_entity


class BuiltinEntity(object):
    def __init__(self, name):
        if not is_builtin_entity(name):
            raise LookupError("Invalid builtin entity {}".format(name))
        self.name = name

    @property
    def json(self):
        return dict()
