import logging

from snips_nlu_ontology import get_ontology_version

from snips_nlu.__about__ import __model_version__, __version__
from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu.pipeline.configs import NLUEngineConfig
from snips_nlu.resources import load_resources

__builtin_entities_version__ = get_ontology_version()
