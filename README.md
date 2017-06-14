# Snips NLU (0.8.2)

[![Build Status](https://jenkins2.snips.ai/buildStatus/icon?job=SDK/snips-nlu/master)](https://jenkins2.snips.ai/job/SDK/job/snips-nlu/view/Branches/job/master)

## Production Use

Python wheels of the `snips-nlu` package can be found on the nexus repository at this URL: https://nexus-repository.snips.ai/#browse/browse/components:pypi-internal

You will need to be signed in to access the repo.


## Development

### Dependencies

You'll need the tensorflow C library installed, refer to the previous section to install it

### Installation
Create a virtual env:

    virtualenv venv

Activate it:

    venv/bin/activate


Update the submodules:

    git submodule update --init --recursive


Then create a pip.conf file within the virtual env:

```
echo "[global]\nindex = https://nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
```

Install the package in edition mode:

    pip install -e .
    

As some dependencies are private, you will need a valid username/password to authenticate to the Nexus repository.

### Initialization

```python
from snips_nlu.nlu_engine import SnipsNLUEngine
```


The NLU Engine can be initialized in two ways:

- You can create an empty engine in order to fit it with a dataset afterwards:
    ```python
    engine = SnipsNLUEngine(language='en')
    ```

- Or you can load an already trained engine:
    ```python
    engine = SnipsNLUEngine.from_dict(engine_as_dict)
    ```

### Serialization
The NLU Engine has an API that allows to persist the object as a dictionary:
```python
engine_as_dict = engine.to_dict()
```

### Parsing
```python
>>> parsing = engine.parse("Turn on the light in the kitchen")
>>> pprint(parsing)
# {
#     "text": "Turn on the light in the kitchen", 
#     "intent": {
#         "intent_name": "switch_light",
#         "probability": 0.95
#     }
#     "slots": [
#         {
#             "value": "on",
#             "range": [5, 7],
#             "slot_name": "on_off",
#         },
#         {
#             "value": "kitchen",
#             "range": [25, 32],
#             "slot_name": "room",
#         }
#     ]
# }
```

### Training
``` python
engine.fitted # False
engine.fit(dataset) 
engine.fitted # True
```

where `dataset` is a dictionary which format is described [here](https://github.com/snipsco/snips-nlu/blob/develop/snips_nlu/tests/resources/sample_dataset.json)
