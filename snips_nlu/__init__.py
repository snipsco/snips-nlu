import builtin_entities_ontology

from snips_nlu.nlu_engine.nlu_engine import SnipsNLUEngine
from snips_nlu.pipeline.configs.nlu_engine import NLUEngineConfig
from snips_nlu.resources import load_resources
from snips_nlu.version import __model_version__, __version__

load_resources()

__builtin_entities_version__ = builtin_entities_ontology.__ontology_version__
