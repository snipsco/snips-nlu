# Custom Intent Parser

## Input data

### Create a dataset from an asset directory

Create an `NLUAssets` directory

#### Ontology

Create a `NLUAssets/mySkill.json` file representing your skill NLU ontology

This file must respect the following structure:

    {
        "intents": [
            {
                "intent": "intent_1",
                "slots": [
                    {"name": "entity_1_name", "entityName": "entity_1"},
                    {"name": "entity_2_name", "entityName": "entity_2"},
                    {"name": "entity_2_other_name", "entityName": "entity_2"}
                    ]
            },
            {
                "intent": "intent_2",
                "slots": []
            }
        ],
        "entities": [
            {
                "entityName": "entity_1",
                "automaticallyExtensible": false,
                "useSynonyms": false
            },
            {
                "entityName": "entity_2",
                "automaticallyExtensible": true,
                "useSynonyms": true
            }
        ]
    }

Make sure that entities names that you use in the intent `slots` description exists in the `entities` list.
 

#### Samples utterances

Place some queries utterances in a the `NLUAssets/SampleUtterances.txt` file.
The file must respect the following format:

    intent_1 an query with a {entity_1_name} and another {entity_2_name} !
    intent_1 another intent with {entity_2_other_name} !
    intent_2 another intent with not entity


#### Entities utterances


You can place utterance of entities in files containing `NLUAssets/<entity_name>.json` files.
If you chose to use `useSynonyms = false` make sure that your file only contains 1 utterance per line.
Otherwise you can define synonyms for a utterance of entity by seperating them with a `;`.

For our example we could put, this content in `NLUAssets/entity_1.json`:
    
    my_entity_1
    my_other_entity_1
    
and the following content in `NLUAssets/entity_2.json`:
    
    my_entity_2;my_entity_2_synonym;my_entity_2_other_synonym
    my_other_entity_2
 
#### Convert your dataset

Convert your text file into a `Dataset` like this:
    
    python custom_intent_parser/data_helpers.py path/to/NLUAssets path/to/dataset_dir
    

Replace the dummy entities in the `dataset_dir/entities/*.json` by real entities utterances 

>>>>>>> rm import persistor

### Use Rasa_nlu backend

The `spacy_sklearn` backend of Rasa_nlu can be used to define an intent parser. Doing so requires the following additional installation steps:

```
pip install spacy
pip install sklearn
pip install scipy
pip install git+https://github.com/golastmile/rasa_nlu.git
python -m spacy.en.download all
```


