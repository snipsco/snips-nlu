import io
import os
import re

from snips_nlu.tokenization import tokenize
from snips_nlu.utils import get_resources_path

CLUSTER_NAMES = ["brown"]

WORD_CLUSTERS = None


def get_word_clusters(language):
    global WORD_CLUSTERS
    word_clusters_paths = dict(
        (name,
         os.path.join(get_resources_path(language), "%s_clusters.txt" % name))
        for name in CLUSTER_NAMES)
    if WORD_CLUSTERS is None:
        WORD_CLUSTERS = dict()
        for name, path in word_clusters_paths.iteritems():
            with io.open(path, encoding="utf8") as f:
                WORD_CLUSTERS[name] = dict()
                for l in f:
                    split = l.rstrip().split("\t")
                    normalized = " ".join(
                        [t.value for t in tokenize(split[0])])
                    if len(split) == 2:
                        WORD_CLUSTERS[name][normalized] = split[1]

    return WORD_CLUSTERS


GAZETTEERS_NAMES = ["top_10000_nouns", "pois", "cities_us", "cities_world",
                    "countries", "regions", "states_us", "stop_words",
                    "street_identifier", "top_10000_words"]

GAZETTEERS = None


def get_gazetteers(language):
    global GAZETTEERS
    gazetteers_paths = dict(
        (name,
         os.path.join(get_resources_path(language), "%s.txt" % name))
        for name in GAZETTEERS_NAMES)
    if GAZETTEERS is None:
        GAZETTEERS = dict()
        for name, path in gazetteers_paths.iteritems():
            with io.open(path, encoding="utf8") as f:
                GAZETTEERS[name] = set()
                for l in f:
                    stripped = l.strip()
                    if len(stripped) > 0:
                        normalized = " ".join(
                            [t.value for t in tokenize(stripped)])
                        GAZETTEERS[name].add(normalized)

    return GAZETTEERS


def get_gazetteer(language, gazetteer_name):
    return get_gazetteers(language)[gazetteer_name]


GAZETTEERS_REGEXES = None


def get_gazetteers_regexes():
    global GAZETTEERS_REGEXES
    if GAZETTEERS_REGEXES is None:
        GAZETTEERS_REGEXES = dict()
        for name, expression_set in GAZETTEERS.iteritems():
            pattern = r"|".join(
                re.escape(e) for e
                in sorted(expression_set, key=len, reverse=True))
            regex = re.compile(pattern, re.IGNORECASE)
            GAZETTEERS_REGEXES[name] = regex
    return GAZETTEERS_REGEXES


def get_gazetteer_regex(gazetteer_name):
    return get_gazetteers_regexes()[gazetteer_name]
