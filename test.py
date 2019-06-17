import io
import json

from snips_nlu.common.utils import unicode_string
from snips_nlu.intent_classifier.ood_detector import OODDetector

with io.open("sample_datasets/lights_dataset.json", encoding="utf8") as f:
    dataset = json.load(f)

detector = OODDetector().fit(dataset)

while True:
    query = input("Enter a query: ").strip()
    if not isinstance(query, str):
        query = query.decode("utf-8")
    query = unicode_string(query)
    print(detector.is_ood(query))
