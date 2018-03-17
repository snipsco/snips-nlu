from __future__ import unicode_literals

from snips_nlu import SnipsNLUEngine
from snips_nlu.pipeline.configs.nlu_engine import DENLUEngineConfig


class DESnipsNLUEngine(SnipsNLUEngine):
    unit_name = "de_nlu_engine"
    config_type = DENLUEngineConfig
