# Snips NLU

Snips NLU (Natural Language understanding) is a python library for extracting meaning from text.

Here is an example of a possible use case:

Input:
```
    text: Set the light to red in the living room please
```

Extracted information:

```    
    intent: SetLightColor
    slots:
        - color: red
        - room: living room
```

This tools uses machine learning, which means you can train it on a specific corpus and Snips will be able to parse out of corpus sentences.

## Production Use

Python wheels of the `snips-nlu` package can be found on the nexus repository at this URL: https://nexus-repository.snips.ai/#browse/browse/components:pypi-internal

You will need to be signed in to access the repo.

## Development

Create a virtual env:

    virtualenv venv


Activate it:

    . venv/bin/activate


Update the submodules:

    git submodule update --init --recursive


Then create a pip.conf file within the virtual env (replacing `<username>` and `<password>` with appropriate values):

    echo "[global]\nindex = https://<username>:<password>@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://<username>:<password>@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf


Install the package in edition mode:

    pip install -e ".[test]"
    

## API

### Data
The input of the NLU engine training is a dataset which format can be found [here](https://github.com/snipsco/snips-nlu/blob/master/samples/sample_dataset.json)

### Code

```python
import io
import json
from pprint import pprint

from snips_nlu import SnipsNLUEngine

############## Initialization ##############

# The nlu config is optional here
with io.open("config.json") as f:
    nlu_config = json.load(f)

engine = SnipsNLUEngine(config=nlu_config)


############## Training ####################

with io.open("path/to/dataset.json") as f:
    dataset = json.load(f)
    
engine.fit(dataset)

############## Parsing #####################

parsing = engine.parse("Turn on the light in the kitchen")
pprint(parsing)
# {
#     "text": "Turn on the light in the kitchen", 
#     "intent": {
#         "intent_name": "switch_light",
#         "probability": 0.95
#     }
#     "slots": [
#         {
#             "value": "on"
#             "range": [5, 7],
#             "slot_name": "light_on_off",
#         },
#         {
#             "value": "kitchen"
#             "range": [25, 32],
#             "slot_name": "light_room",
#         }
#     ]
# }


############## Serialization ###############

with io.open("path/to/trained_engine.json", mode="w", encoding="utf8") as f:
    f.write(json.dumps(engine.to_dict()).decode())

############## Deserialization #############

with io.open("path/to/trained_engine.json") as f:
    trained_engine_dict = json.load(f)
    
trained_engine = SnipsNLUEngine.from_dict(trained_engine_dict)
```

### CLI

```bash
train-engine /path/to/input/dataset.json /path/to/output/trained_engine.json
engine-inference /path/to/output/trained_engine.json
```

### Versioning
The NLU Engine has a separate versioning for the underlying model:
``` python
import snips_nlu

model_version = snips_nlu.__model_version__
python_package_version = snips_nlu.__version__
```


## Test coverage

```bash
venv/bin/coverage run --source=snips_nlu -m unittest discover -s snips_nlu/tests/
venv/bin/coverage report -m
```