# Custom Intent Parser

## Input data

### Create a dataset from a text file

The simplest way to create data is to put them in the following format in a `dataset.txt` file:


    ExampleIntent Place your @entity_name_1 like this and add a role like that @entity_name_2:role
    BookTaxi Book me a @transport_name to go from @location:from to @location:to for @number_of_persons
 
Convert your text file into a `Dataset` like this:
    
    python custom_intent_parser/data_helpers.py path/to/dataset.txt path/to/dataset_dir
    
Replace the dummy entities in the `dataset_dir/entities/*.json` by real entities utterances 

### Use Rasa_nlu backend

The `spacy_sklearn` backend of Rasa_nlu can be used to define an intent parser. Doing so requires the following additional installation steps:

```
pip install spacy
pip install sklearn
pip install scipy
pip install git+https://github.com/golastmile/rasa_nlu.git
python -m spacy.en.download all
```

