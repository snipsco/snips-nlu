from semantic_version import Version

from snips_nlu import __version__, __model_version__
from snips_nlu.tests.utils import SnipsTest


class TestVersion(SnipsTest):
    def test_version_should_be_semantic(self):
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
        self.assertTrue(valid, "Version number '%s' is not semantically valid"
                        % version)

    def test_model_version_should_be_semantic(self):
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
        self.assertTrue(valid, "Version number '%s' is not semantically valid"
                        % model_version)
