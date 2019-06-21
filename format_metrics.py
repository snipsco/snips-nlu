import io
import json
import sys


def format_metrics_for_console(metrics_):
    for intent_metrics in metrics_["metrics"].values():
        if "slots" not in intent_metrics:
            intent_metrics["slots"] = dict()
    return {
        "metrics_type": "nlu",
        "config": {
            "assistantId": "proj_60YKe46pYB0",
            "nbFolds": 5,
            "trainingUtterances": [1]
        },
        "results": [
            {
                "config": {
                    "nb_folds": 5,
                    "train_size_ratio": 1,
                    "intents_filter": None,
                    "assistant_id": None,
                    "training_version": None,
                    "inference_version": None,
                },
                "metrics": metrics_["metrics"],
                "parsing_errors": metrics_["parsing_errors"],
                "confusion_matrix": metrics_["confusion_matrix"],
            }
        ],
        "createdAt": {"$date": 0},
        "jobId": "0"
    }


if __name__ == "__main__":
    metrics_path = sys.argv[1]
    with io.open(metrics_path, encoding="utf8") as f:
        metrics = json.load(f)
    print(json.dumps(format_metrics_for_console(metrics), indent=2,
                     sort_keys=True))
