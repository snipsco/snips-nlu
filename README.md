# Snips NLU


## Dependencies

#### 1. Rust

You need to have [Rust](https://www.rust-lang.org/en-US/install.html) installed, then do:

#### 2. Snips Tensorflow build

Follow installation instructions [here](https://github.com/snipsco/tensorflow-build).


## Installation
Create a virtual env
    
    virtualenv venv    

Activate it
    
    venv/bin/activate

Install the package
    
    pip install -f https://nexus-repository.snips.ai/repository/pypi-internal/packages/snips-queries/0.1.0a1/snips_queries-0.1.0a1-py2.py3-none-macosx_10_11_x86_64.whl -e .

As the repository is private, you will need a valid username/password to authenticate to the Nexus repository.

### Initialization

```python
from snips_nlu.nlu_engine import SnipsNLUEngine
```


The NLU Engine can be initialized in two ways:

- You can create an empty engine in order to fit it with a dataset afterwards:
    ```python
    engine = SnipsNLUEngine()
    ```

- Or you can load an already trained engine from a dictionary:
    ```python
    engine = SnipsNLUEngine.from_dict(engine_dict)
    ```

### Serialization
The NLU Engine has an API that allows to persist the object as a dictionary:
```python
engine_dict = engine.to_dict()
```

### Parsing
```python
>>> parsing = engine.parse("Turn on the light in the kitchen")
>>> pprint(parsing)
# {
#     "text": "Turn on the light in the kitchen", 
#     "intent": {
#         "name": "switch_light",
#         "prob": 0.95
#     }
#     "slots": [
#         {
#             "value": "on",
#             "range": [5, 7],
#             "name": "on_off",
#         },
#         {
#             "value": "kitchen",
#             "range": [25, 32],
#             "name": "room",
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
where `dataset` is a dictionary with the following format:
```python
dataset = 
    {
        "intents": {
            "dummy_intent_1": {
                "utterances": [
                    {
                        "data": [
                            {
                                "text": "This is a "
                            },
                            {
                                "text": "dummy_1",
                                "entity": "dummy_entity_1",
                                "slot_name": "dummy_slot_name"
                            },
                            {
                                "text": " query."
                            }
                        ]
                    },
                    {
                        "data": [
                            {
                                "text": "This is another "
                            },
                            {
                                "text": "dummy_2",
                                "entity": "dummy_entity_2",
                                "slot_name": "dummy_slot_name2"
                            },
                            {
                                "text": " query."
                            }
                        ]
                    }
                ]
            }
        },
        "entities": {
            "dummy_entity_1": {
                "use_synonyms": True,
                "automatically_extensible": False,
                "data": [
                    {
                        "value": "dummy1",
                        "synonyms": [
                            "dummy1",
                            "dummy1_bis"
                        ]
                    },
                    {
                        "value": "dummy2",
                        "synonyms": [
                            "dummy2",
                            "dummy2_bis"
                        ]
                    }
                ]
            },
            "dummy_entity_2": {
                "use_synonyms": False,
                "automatically_extensible": True,
                "data": [
                    {
                        "value": "dummy2",
                        "synonyms": [
                            "dummy2"
                        ]
                    }
                ]
            }
        }
    }
```
    