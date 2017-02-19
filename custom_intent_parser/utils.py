import copy
import json

from collections import OrderedDict


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


def sequence_equal(seq, other_seq):
    return len(seq) == len(other_seq) and sorted(seq) == sorted(other_seq)


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        if "size_limit" not in kwds:
            raise ValueError("'size_limit' must be passed as a keywords "
                             "argument")
        self.size_limit = kwds.pop("size_limit")
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        if len(args) == 1 and len(args[0]) + len(kwds) > self.size_limit:
            raise ValueError("Tried to initialize LimitedSizedDict with more "
                             "value than permitted with 'limit_size'")
        super(LimitedSizeDict, self).__init__(*args, **kwds)

    def __setitem__(self, key, value, dict_setitem=OrderedDict.__setitem__):
        dict_setitem(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def transform_to_rasa_format(dataset):
    queries=dataset.queries
    if len(queries.keys())>1:
        raise ValueError("Training data should only contain data about a single intent")

    intent=queries.keys()[0]

    rasa_dataset=[]
    for query in queries[intent]:
        text="".join(map(lambda span: span["text"], query["data"]))
        spans_with_entity=filter(lambda span: span.get("entity", None), query["data"])
        entities=[]
        current_index=0
        for span in query["data"]:
            if span.get("entity", None) is not None:
                entities.append({
                    "start": current_index,
                    "end": current_index + len(span["text"]),
                    "value": span["text"],
                    "entity": span.get("role", span.get("entity"))
                })
            current_index+=len(span["text"])
        rasa_dataset.append({
            "text": text,
            "intent": intent,
            "entities": entities
        })

    with open('custom_intent_parser/tests/rasa_nlu_test_data.json') as data_file:    
        rasa_test_data = json.load(data_file)

    # TODO: improve this add noise as Other utterances, or even better adversarial data 
    all_queries_as_text=[query["text"] for query in rasa_dataset]
    for example in rasa_test_data["rasa_nlu_data"]['entity_examples']:
        if example['text'] not in all_queries_as_text:
            rasa_dataset.append(example)


    rasa_dataset_without_entities=copy.deepcopy(rasa_dataset)
    for example in rasa_dataset_without_entities: 
        del example['entities']

    return rasa_test_data


    return {"rasa_nlu_data": 
        { 
            "entity_examples": rasa_dataset,
            "intent_examples": rasa_dataset_without_entities
        }
    }