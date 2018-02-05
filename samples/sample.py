from __future__ import unicode_literals

import io
import json

from snips_nlu import SnipsNLUEngine

with io.open("sample_dataset.json") as f:
    sample_dataset = json.load(f)

with io.open("configs/config_en.json") as f:
    config = json.load(f)

nlu_engine = SnipsNLUEngine(config=config)
nlu_engine.fit(sample_dataset)

print(nlu_engine.parse("Could you please turn the lights on in the kitchen ?"))
