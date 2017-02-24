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
    
    python setup.py install


## Alexa style data format

### Create a dataset from an asset directory

Create an `nluAsset` directory

#### Ontology

Create a `nluAsset/mySkill.json` file representing your skill NLU ontology

This file must respect the following structure:

    {
        "intents": [
            {
                "intent": "intent_1",
                "slots": [
                    {"slotName": "slot_1", "entity": "entity_1"},
                    {"slotName": "slot_2", "entity": "entity_2"},
                    {"slotName": "slot_3", "entity": "entity_2"}
                    ]
            },
            {
                "intent": "intent_2",
                "slots": []
            }
        ],
        "entities": [
            {
                "entity": "entity_1",
                "automaticallyExtensible": false,
                "useSynonyms": false
            },
            {
                "entity": "entity_2",
                "automaticallyExtensible": true,
                "useSynonyms": true
            }
        ]
    }

Make sure that entities names that you use in the intent `slots` description exists in the `entities` list.
 

#### Samples utterances

Place some queries utterances in a the `nluAsset/SampleUtterances.txt` file.
The file must respect the following format:

    intent_1 an query with a {entity_1_name} and another {entity_2_name} !
    intent_1 another intent with {entity_2_other_name} !
    intent_2 another intent with not entity


#### Entities utterances


You can place utterance of entities in files containing `nluAsset/<entity_name>.txt` files.
If you chose to use `useSynonyms = false` make sure that your file only contains 1 utterance per line.
Otherwise you can define synonyms for a utterance of entity by seperating them with a `;`.

For our example we could put, this content in `nluAsset/entity_1.txt`:
    
    my_entity_1
    my_other_entity_1
    
and the following content in `nluAsset/entity_2.txt`:
    
    my_entity_2;my_entity_2_synonym;my_entity_2_other_synonym
    my_other_entity_2
 
#### Convert your dataset

Convert your text file into a `Dataset` like this:
    
    python custom_intent_parser/data_helpers.py path/to/nluAsset path/to/dataset_dir
    