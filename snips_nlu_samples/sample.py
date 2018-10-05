from __future__ import unicode_literals, print_function

import json
from pathlib import Path

from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.default_configs import CONFIG_EN

SAMPLE_DATASET_PATH = Path(__file__).parent / "sample_dataset.json"

with SAMPLE_DATASET_PATH.open(encoding="utf8") as f:
    sample_dataset = json.load(f)

load_resources("en")
nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
nlu_engine.fit(sample_dataset)

text = "What will be the weather in San Francisco next week?"
parsing = nlu_engine.parse(text)
print(json.dumps(parsing, indent=2))
