import copy
import json


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
