from snips_nlu_ontology import get_ontology_version

from snips_nlu.nlu_engine import SnipsNLUEngine, DESnipsNLUEngine
from snips_nlu.pipeline.configs import NLUEngineConfig
from snips_nlu.resources import load_resources
from snips_nlu.version import __model_version__, __version__

__builtin_entities_version__ = get_ontology_version()
