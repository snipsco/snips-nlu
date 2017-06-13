from __future__ import unicode_literals

import argparse
import io
import json
import os

from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_functions import crf_features
from snips_nlu.tests.utils import TEST_PATH
from snips_nlu.tokenization import tokenize


def export_feature(language_code, output_path):
    sample_path = os.path.join(TEST_PATH, "resources",
                               "sample_sentences_%s.txt" % language_code)
    with io.open(sample_path, encoding="utf8") as f:
        sentences = [l.strip() for l in f]

    sentences = [s for s in sentences if len(s) > 0]
    language = Language.from_iso_code(language_code)
    tagging_scheme = TaggingScheme.BIO
    intent_custom_entities = dict()
    features = crf_features(intent_custom_entities, language)
    tagger = CRFTagger(default_crf_model(), features, tagging_scheme, language)

    features = dict()
    for sentence in sentences:
        features[sentence] = tagger.compute_features(tokenize(sentence))

    with io.open(output_path, "w", encoding="utf8") as f:
        data = json.dumps(features, indent=2).decode("utf8")
        f.write(data)


def main_export_ontology():
    parser = argparse.ArgumentParser()
    parser.add_argument("language_code", type=unicode)
    parser.add_argument("output_path", type=unicode)
    export_feature(**vars(parser.parse_args()))


if __name__ == "__main__":
    main_export_ontology()
