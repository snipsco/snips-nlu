# coding=utf-8
import unittest

from snips_nlu.config import NLUConfig


class TestConfig(unittest.TestCase):
    def test_nlu_config_from_dic(self):
        # Given
        config = NLUConfig()

        # When
        config_as_dict = config.to_dict()
        new_config = NLUConfig.from_dict(config_as_dict)

        # Then
        self.assertEqual(config, new_config)


if __name__ == '__main__':
    unittest.main()
