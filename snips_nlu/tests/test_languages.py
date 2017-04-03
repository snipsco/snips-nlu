import unittest

from snips_nlu.languages import Language


class TestLanguage(unittest.TestCase):
    def test_iso_unique(self):
        # Given
        language = Language

        # When
        nb_lang = len(set([lang.value['iso'] for lang in language]))

        # Then
        expected_nb_lang = len(Language)

        self.assertEqual(nb_lang, expected_nb_lang)
