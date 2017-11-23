import argparse
import io
import json
import os
from pprint import pprint


def main_train_engine():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=unicode)
    parser.add_argument("dataset_path", type=unicode)
    parser.add_argument("output_path", type=unicode)
    parser.add_argument("--config-path", type=unicode)
    args = vars(parser.parse_args())

    from snips_nlu import SnipsNLUEngine
    from snips_nlu.config import NLUConfig

    dataset_path = args.pop("dataset_path")
    with io.open(dataset_path, "r", encoding="utf8") as f:
        dataset = json.load(f)

    if args.get("config_path") is not None:
        config_path = args.pop("config_path")
        with io.open(config_path, "r", encoding="utf8") as f:
            config = json.load(f)
    else:
        config = NLUConfig()

    language = args.pop("language")
    engine = SnipsNLUEngine(language, config).fit(dataset)
    print "Create and train the engine..."

    output_path = args.pop("output_path")
    serialized_engine = json.dumps(engine.to_dict()).decode("utf8")
    with io.open(output_path, "w", encoding="utf8") as f:
        f.write(serialized_engine)
    print "Saved the trained engine to %s" % output_path


def main_engine_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_path", type=unicode)
    args = vars(parser.parse_args())

    from snips_nlu import SnipsNLUEngine

    training_path = args.pop("training_path")
    with io.open(os.path.abspath(training_path), "r", encoding="utf8") as f:
        engine_dict = json.load(f)
    engine = SnipsNLUEngine.from_dict(engine_dict)

    while True:
        query = raw_input("Enter a query (type 'q' to quit): ").strip()
        if isinstance(query, str):
            query = query.decode("utf8")
        if query == "q":
            break
        pprint(engine.parse(query))


def main_cross_val_metrics():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=unicode)
    parser.add_argument("output_path", type=unicode)
    parser.add_argument("--nb-folds", type=int)
    parser.add_argument("--train-size-ratio", type=float)
    parser.add_argument("--include-errors", type=bool)

    args = vars(parser.parse_args())

    from snips_nlu import SnipsNLUEngine as TrainingEngine
    from snips_nlu_rust import NLUEngine as InferenceEngine
    from nlu_metrics import compute_cross_val_nlu_metrics

    dataset_path = args.pop("dataset_path")
    output_path = args.pop("output_path")

    def progression_handler(progress):
        print "%d%%" % int(progress * 100)

    metrics_args = dict(
        dataset=dataset_path,
        training_engine_class=TrainingEngine,
        inference_engine_class=InferenceEngine,
        progression_handler=progression_handler
    )
    if args.get("nb-folds") is not None:
        nb_folds = args.pop("nb-folds")
        metrics_args.update(dict(nb_folds=nb_folds))
    if args.get("train-size-ratio") is not None:
        train_size_ratio = args.pop("train-size-ratio")
        metrics_args.update(dict(train_size_ratio=train_size_ratio))
    include_errors = False
    if args.get("include-errors") is not None:
        include_errors = True

    metrics = compute_cross_val_nlu_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("errors")

    with io.open(output_path, mode="w") as f:
        f.write(json.dumps(metrics).decode())


def main_train_test_metrics():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset_path", type=unicode)
    parser.add_argument("test_dataset_path", type=unicode)
    parser.add_argument("output_path", type=unicode)
    parser.add_argument("--include-errors", type=bool)

    args = vars(parser.parse_args())

    from snips_nlu import SnipsNLUEngine as TrainingEngine
    from snips_nlu_rust import NLUEngine as InferenceEngine
    from nlu_metrics import compute_train_test_nlu_metrics

    train_dataset_path = args.pop("train_dataset_path")
    test_dataset_path = args.pop("test_dataset_path")
    output_path = args.pop("output_path")

    metrics_args = dict(
        train_dataset=train_dataset_path,
        test_dataset=test_dataset_path,
        training_engine_class=TrainingEngine,
        inference_engine_class=InferenceEngine,
    )

    include_errors = False
    if args.get("include_errors") is not None:
        include_errors = True

    metrics = compute_train_test_nlu_metrics(**metrics_args)
    if not include_errors:
        metrics.pop("errors")

    with io.open(output_path, mode="w") as f:
        f.write(json.dumps(metrics).decode())
