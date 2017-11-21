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
