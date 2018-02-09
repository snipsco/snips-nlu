import builtin_entities_ontology as beo

ONTOLOGY = beo.get_ontology()
BUILTIN_ENTITIES = set(e['label'] for e in ONTOLOGY['entities'])
