from __future__ import unicode_literals

from snips_nlu.constants import (
    DATA_PATH, STOP_WORDS, CUSTOM_ENTITY_PARSER_USAGE, STEMS, WORD_CLUSTERS,
    GAZETTEERS, NOISE)
from snips_nlu.entity_parser import CustomEntityParserUsage
from snips_nlu.resources import (
    MissingResource, _get_resource, load_resources, _persist_stop_words,
    _load_stop_words, _load_noise, _persist_noise, _load_word_clusters,
    _persist_word_clusters, _load_gazetteer, _persist_gazetteer, _load_stems,
    _persist_stems, get_stems)
from snips_nlu.tests.utils import FixtureTest


class TestResources(FixtureTest):
    def test_should_load_resources_from_data_path(self):
        # When
        resources = load_resources("en")

        # Then
        self.assertGreater(len(get_stems(resources)), 0)

    def test_should_load_resources_with_requirements(self):
        # Given
        required_resources = {
            STOP_WORDS: False,
            CUSTOM_ENTITY_PARSER_USAGE: CustomEntityParserUsage.WITHOUT_STEMS,
            STEMS: False,
            NOISE: True,
            WORD_CLUSTERS: ["brown_clusters"],
            GAZETTEERS: ["top_10000_words"],
        }

        # When
        resources = load_resources("en", required_resources)

        # Then
        self.assertListEqual(["top_10000_words"], list(resources[GAZETTEERS]))
        self.assertListEqual(["brown_clusters"],
                             list(resources[WORD_CLUSTERS]))
        self.assertIsNone(resources[STOP_WORDS])
        self.assertIsNone(resources[STEMS])
        self.assertGreater(len(resources[NOISE]), 0)

    def test_should_not_load_resources_with_bad_requirements(self):
        # When / Then
        with self.assertRaises(ValueError) as cm:
            load_resources("en", {GAZETTEERS: ["unknown"]})
        self.assertTrue("Unknown gazetteer" in str(cm.exception))

        # When / Then
        with self.assertRaises(ValueError) as cm:
            load_resources("en", {WORD_CLUSTERS: ["unknown"]})
        self.assertTrue("Unknown word clusters" in str(cm.exception))

    def test_should_load_resources_from_package(self):
        # When
        resources = load_resources("snips_nlu_en")

        # Then
        self.assertGreater(len(get_stems(resources)), 0)

    def test_should_load_resources_from_path(self):
        # Given
        resources_path = DATA_PATH / "en"

        # When
        resources = load_resources(str(resources_path))

        # Then
        self.assertGreater(len(get_stems(resources)), 0)

    def test_should_fail_loading_unknown_resources(self):
        # Given
        unknown_resource_name = "foobar"

        # When / Then
        with self.assertRaises(MissingResource):
            load_resources(unknown_resource_name)

    def test_should_raise_missing_resource_when_resource_not_found(self):
        # Given
        resources = dict()

        # When
        with self.assertRaises(MissingResource):
            _get_resource(resources, "foobar")

    def test_should_persist_stop_words(self):
        # Given
        stop_words = _load_stop_words(DATA_PATH / "en" / "stop_words.txt")

        # When
        _persist_stop_words(stop_words, self.fixture_dir / "stop_words.txt")
        loaded_stop_words = _load_stop_words(
            self.fixture_dir / "stop_words.txt")

        # Then
        self.assertSetEqual(stop_words, loaded_stop_words)

    def test_should_persist_noise(self):
        # Given
        noise = _load_noise(DATA_PATH / "en" / "noise.txt")

        # When
        _persist_noise(noise, self.fixture_dir / "noise.txt")
        loaded_noise = _load_noise(self.fixture_dir / "noise.txt")

        # Then
        self.assertListEqual(noise, loaded_noise)

    def test_should_persist_word_clusters(self):
        # Given
        word_clusters = _load_word_clusters(
            DATA_PATH / "en" / "word_clusters" / "brown_clusters.txt")

        # When
        _persist_word_clusters(word_clusters,
                               self.fixture_dir / "brown_clusters.txt")
        loaded_word_clusters = _load_word_clusters(
            self.fixture_dir / "brown_clusters.txt")

        # Then
        self.assertDictEqual(word_clusters, loaded_word_clusters)

    def test_should_persist_gazetteer(self):
        # Given
        gazetteer = _load_gazetteer(
            DATA_PATH / "en" / "gazetteers" / "top_10000_words.txt")

        # When
        _persist_gazetteer(gazetteer, self.fixture_dir / "top_10000_words.txt")
        loaded_gazetteer = _load_gazetteer(
            self.fixture_dir / "top_10000_words.txt")

        # Then
        self.assertSetEqual(gazetteer, loaded_gazetteer)

    def test_should_persist_stems(self):
        # Given
        stems = _load_stems(DATA_PATH / "en" / "stemming" / "stems.txt")

        # When
        _persist_stems(stems, self.fixture_dir / "stems.txt")
        loaded_stems = _load_stems(self.fixture_dir / "stems.txt")

        # Then
        self.assertDictEqual(stems, loaded_stems)
