from __future__ import unicode_literals

import json
from builtins import next
from pathlib import Path

from snips_nlu.constants import (STOP_WORDS, WORD_CLUSTERS, GAZETTEERS, NOISE,
                                 STEMS, DATA_PATH, RESOURCES_DIR)
from snips_nlu.utils import is_package, get_package_path

_RESOURCES = dict()


class MissingResource(LookupError):
    pass


def clear_resources():
    _RESOURCES.clear()


def load_resources(name):
    """Load language specific resources

    Args:
        name (str): Resource name as in ``snips-nlu download <name>``. Can also
            be the name of a python package or a directory path.

    Note:
        Language resources must be loaded before fitting or parsing
    """
    if name in set(d.name for d in DATA_PATH.iterdir()):
        load_resources_from_dir(DATA_PATH / name)
    elif is_package(name):
        package_path = get_package_path(name)
        resources_sub_dir = get_resources_sub_directory(package_path)
        load_resources_from_dir(resources_sub_dir)
    elif Path(name).exists():
        path = Path(name)
        if (path / "__init__.py").exists():
            path = get_resources_sub_directory(path)
        load_resources_from_dir(path)
    else:
        raise MissingResource("Language resource '{r}' not found. This may be "
                              "solved by running "
                              "'python -m snips_nlu download {r}'"
                              .format(r=name))


def load_resources_from_dir(resources_dir):
    with (resources_dir / "metadata.json").open() as f:
        metadata = json.load(f)
    language = metadata["language"]
    if language in _RESOURCES:
        return

    gazetteer_names = metadata["gazetteers"]
    gazetteers = _load_gazetteers(resources_dir / "gazetteers",
                                  gazetteer_names)
    stems = _load_stems(resources_dir / "stemming", metadata["stems"])
    clusters_names = metadata["word_clusters"]
    word_clusters = _load_word_clusters(resources_dir / "word_clusters",
                                        clusters_names)
    stop_words = _load_stop_words(resources_dir, metadata["stop_words"])
    noise = _load_noise(resources_dir, metadata["noise"])

    _RESOURCES[language] = {
        WORD_CLUSTERS: word_clusters,
        GAZETTEERS: gazetteers,
        STOP_WORDS: stop_words,
        NOISE: noise,
        STEMS: stems,
        RESOURCES_DIR: str(resources_dir),
    }


def get_resources_sub_directory(resources_dir):
    resources_dir = Path(resources_dir)
    with (resources_dir / "metadata.json").open() as f:
        metadata = json.load(f)
    resource_name = metadata["name"]
    version = metadata["version"]
    sub_dir_name = "{r}-{v}".format(r=resource_name, v=version)
    return resources_dir / sub_dir_name


def resource_exists(language, resource_name):
    """Tell if the resource specified by the resource_name exist

        Args:
            language (str): language
            resource_name (str): the resource name
        Returns:
            bool: whether the resource exists or not
    """
    return resource_name in _RESOURCES[language] \
           and _RESOURCES[language][resource_name] is not None


def get_stop_words(language):
    return _get_resource(language, STOP_WORDS)


def get_noise(language):
    return _get_resource(language, NOISE)


def get_word_clusters(language):
    return _get_resource(language, WORD_CLUSTERS)


def get_word_cluster(language, cluster_name):
    word_clusters = get_word_clusters(language)
    if cluster_name not in word_clusters:
        raise MissingResource("Word cluster '{}' not found for language '{}'"
                              .format(cluster_name, language))
    return word_clusters[cluster_name]


def get_gazetteer(language, gazetteer_name):
    gazetteers = _get_resource(language, GAZETTEERS)
    if gazetteer_name not in gazetteers:
        raise MissingResource("Gazetteer '{}' not found for language '{}'"
                              .format(gazetteer_name, language))
    return gazetteers[gazetteer_name]


def get_stems(language):
    return _get_resource(language, STEMS)


def get_resources_dir(language):
    return _get_resource(language, RESOURCES_DIR)


def _get_resource(language, resource_name):
    if language not in _RESOURCES:
        raise MissingResource(
            "Missing resources for language '%s', please load them with the "
            "load_resources function" % language)
    if resource_name not in _RESOURCES[language] \
            or _RESOURCES[language][resource_name] is None:
        raise MissingResource("Resource '{}' not found for language '{}'"
                              .format(resource_name, language))
    return _RESOURCES[language][resource_name]


def _load_stop_words(resources_dir, stop_words_filename):
    if not stop_words_filename:
        return None
    stop_words_path = (resources_dir / stop_words_filename).with_suffix(".txt")
    with stop_words_path.open(encoding='utf8') as f:
        stop_words = set(l.strip() for l in f)
    return stop_words


def _load_noise(resources_dir, noise_filename):
    if not noise_filename:
        return None
    noise_path = (resources_dir / noise_filename).with_suffix(".txt")
    with noise_path.open(encoding='utf8') as f:
        # Here we split on a " " knowing that it's always ignored by
        # the tokenization (see tokenization unit tests)
        # It is not important to tokenize precisely as this noise is just used
        # to generate utterances for the None intent
        noise = next(f).split()
    return noise


def _load_word_clusters(word_clusters_dir, clusters_names):
    if not clusters_names:
        return dict()

    clusters = dict()
    for clusters_name in clusters_names:
        clusters_path = (word_clusters_dir / clusters_name).with_suffix(".txt")
        clusters[clusters_name] = dict()
        with clusters_path.open(encoding="utf8") as f:
            for line in f:
                split = line.rstrip().split("\t")
                clusters[clusters_name][split[0]] = split[1]
    return clusters


def _load_gazetteers(gazetteers_dir, gazetteer_names):
    if not gazetteer_names:
        return dict()

    gazetteers = dict()
    for gazetteer_name in gazetteer_names:
        gazetteer_path = (gazetteers_dir / gazetteer_name).with_suffix(".txt")
        with gazetteer_path.open(encoding="utf8") as f:
            gazetteers[gazetteer_name] = set(v.strip() for v in f)
    return gazetteers


def _load_stems(stems_dir, filename):
    if not filename:
        return None
    stems_path = (stems_dir / filename).with_suffix(".txt")
    stems = dict()
    with stems_path.open(encoding="utf8") as f:
        for line in f:
            elements = line.strip().split(',')
            stem = elements[0]
            for value in elements[1:]:
                stems[value] = stem
    return stems
