from __future__ import unicode_literals, print_function

import io
import json
from os.path import dirname, abspath, join

from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.default_configs import CONFIG_EN

SAMPLE_DATASET_PATH = join(dirname(abspath(__file__)), "sample_dataset.json")

with io.open(SAMPLE_DATASET_PATH) as f:
    sample_dataset = json.load(f)

load_resources("en")
nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
nlu_engine.fit(sample_dataset)

text = "What will be the weather in San Francisco next week?"
parsing = nlu_engine.parse(text)
print(json.dumps(parsing, indent=2))
