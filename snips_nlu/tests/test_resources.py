from __future__ import unicode_literals

from mock import patch

from snips_nlu.constants import DATA_PATH
from snips_nlu.resources import (
    MissingResource, _RESOURCES, _get_resource, clear_resources,
    load_resources, _persist_stop_words, _load_stop_words, _load_noise,
    _persist_noise, _load_word_clusters, _persist_word_clusters,
    _load_gazetteer, _persist_gazetteer, _load_stems, _persist_stems)
from snips_nlu.tests.utils import FixtureTest


class TestResources(FixtureTest):
    def test_should_load_resources_from_data_path(self):
        # Given
        clear_resources()

        # When
        load_resources("en")

        # Then
        self.assertTrue(resource_exists("en", "gazetteers"))

    def test_should_load_resources_from_package(self):
        # Given
        clear_resources()

        # When
        load_resources("snips_nlu_en")

        # Then
        self.assertTrue(resource_exists("en", "gazetteers"))

    def test_should_load_resources_from_path(self):
        # Given
        clear_resources()
        resources_path = DATA_PATH / "en"

        # When
        load_resources(str(resources_path))

        # Then
        self.assertTrue(resource_exists("en", "gazetteers"))

    def test_should_fail_loading_unknown_resources(self):
        # Given
        unknown_resource_name = "foobar"

        # When / Then
        with self.assertRaises(MissingResource):
            load_resources(unknown_resource_name)

    def test_should_raise_missing_resource_when_language_not_found(self):
        # Given
        mocked_value = dict()

        # When
        with patch("snips_nlu.resources._RESOURCES", mocked_value):
            with self.assertRaises(MissingResource):
                _get_resource("en", "foobar")

    def test_should_raise_missing_resource_when_resource_not_found(self):
        # Given
        mocked_value = {"en": dict()}

        # When
        with patch("snips_nlu.resources._RESOURCES", mocked_value):
            with self.assertRaises(MissingResource):
                _get_resource("en", "foobar")

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


def resource_exists(language, resource_name):
    return resource_name in _RESOURCES[language] \
           and _RESOURCES[language][resource_name] is not None
