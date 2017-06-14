import unittest

from semantic_version import Version

from snips_nlu import __version__, __model_version__


class TestVersion(unittest.TestCase):
    def version_should_be_semantic(self):
        # Given
        version = __version__

        # When
        valid = False
        try:
            Version(version)
            valid = True
        except ValueError:
            pass

        # Then
        self.assertTrue(valid, "Version number '%s' is not semantically valid")

    def model_version_should_be_semantic(self):
        # Given
        model_version = __model_version__

        # When
        valid = False
        try:
            Version(model_version)
            valid = True
        except ValueError:
            pass

        # Then
        self.assertTrue(valid, "Version number '%s' is not semantically valid")
