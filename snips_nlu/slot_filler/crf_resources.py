import io
import os
import re

from snips_nlu.tokenization import tokenize

RESOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "resources")
CLUSTER_NAMES = ["brown"]
WORD_CLUSTERS_PATHS = dict(
    (name, os.path.join(RESOURCE_PATH, "%s_clusters.txt" % name))
    for name in CLUSTER_NAMES)

WORD_CLUSTERS = None


def get_word_clusters():
    global WORD_CLUSTERS
    if WORD_CLUSTERS is None:
        WORD_CLUSTERS = dict()
        for name, path in WORD_CLUSTERS_PATHS.iteritems():
            with io.open(path, encoding="utf8") as f:
                WORD_CLUSTERS[name] = dict()
                for l in f:
                    split = l.rstrip().split("\t")
                    normalized = " ".join(
                        [t.value for t in tokenize(split[0])])
                    if len(split) == 2:
                        WORD_CLUSTERS[name][normalized] = split[1]

    return WORD_CLUSTERS


GAZETTEERS_NAMES = ["english_top_10000"]
GAZETTEERS_PATHS = dict((name, os.path.join(RESOURCE_PATH, "%s.txt" % name))
                        for name in GAZETTEERS_NAMES)
GAZETTEERS = None


def get_gazetteers():
    global GAZETTEERS
    if GAZETTEERS is None:
        GAZETTEERS = dict()
        for name, path in GAZETTEERS_PATHS.iteritems():
            with io.open(path, encoding="utf8") as f:
                GAZETTEERS[name] = set()
                for l in f:
                    stripped = l.strip()
                    if len(stripped) > 0:
                        normalized = " ".join(
                            [t.value for t in tokenize(stripped)])
                        GAZETTEERS[name].add(normalized)

    return GAZETTEERS


def get_gazetteer(gazetteer_name):
    return get_gazetteers()[gazetteer_name]


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
