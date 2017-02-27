from collections import OrderedDict
import copy
import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
            raise ValueError("'size_limit' must be passed as a keyword "
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

    def __eq__(self, other):
        if self.size_limit != other.size_limit:
            return False
        return super(LimitedSizeDict, self).__eq__(other)


def transform_to_rasa_format(dataset):
    queries = dataset.queries
    
    rasa_dataset = []
    for intent in queries.keys():
        for query in queries[intent]:
            text = "".join(map(lambda span: span["text"], query["data"]))
            spans_with_entity = filter(
                lambda span: span.get("entity", None), query["data"])
            entities = []
            current_index = 0
            for span in query["data"]:
                if span.get("entity", None) is not None:
                    entities.append({
                        "start": current_index,
                        "end": current_index + len(span["text"]),
                        "value": span["text"],
                        "entity": span.get("role", span.get("entity"))
                    })
                current_index += len(span["text"])
            rasa_dataset.append({
                "text": text,
                "intent": intent,
                "entities": entities
            })

    rasa_dataset_without_entities = copy.deepcopy(rasa_dataset)
    for example in rasa_dataset_without_entities:
        del example['entities']

    return {"rasa_nlu_data":
        {
            "entity_examples": rasa_dataset,
            "intent_examples": rasa_dataset_without_entities
        }
    }

