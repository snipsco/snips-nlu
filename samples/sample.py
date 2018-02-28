from __future__ import unicode_literals, print_function

import io
import json

from snips_nlu import SnipsNLUEngine, load_resources

with io.open("sample_dataset.json") as f:
    sample_dataset = json.load(f)

with io.open("configs/config_en.json") as f:
    config = json.load(f)

load_resources(sample_dataset["language"])
nlu_engine = SnipsNLUEngine(config=config)
nlu_engine.fit(sample_dataset)

text = "What will be the weather in San Francisco next week?"
parsing = nlu_engine.parse(text)
print(json.dumps(parsing, indent=2))
