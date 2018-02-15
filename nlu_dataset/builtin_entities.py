from __future__ import unicode_literals

import builtin_entities_ontology as beo
from future.builtins import object

ONTOLOGY = beo.get_ontology()
BUILTIN_ENTITIES = set(e['label'] for e in ONTOLOGY['entities'])


class BuiltinEntity(object):
    def __init__(self, name):
        if name not in BUILTIN_ENTITIES:
            raise LookupError("Invalid builtin entity {}".format(name))
        self.name = name

    @property
    def json(self):
        return dict()
