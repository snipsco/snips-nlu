# coding=utf-8
from __future__ import unicode_literals

import shutil
import tempfile

from snips_nlu import SnipsNLUEngine
from snips_nlu.cli import cross_val_metrics, parse, train, train_test_metrics
from snips_nlu.tests.utils import BEVERAGE_DATASET_PATH, SnipsTest, TEST_PATH


def mk_sys_argv(args):
    return ["program_name"] + args


class TestCLI(SnipsTest):
    fixture_dir = TEST_PATH / "cli_fixture"

    # pylint: disable=protected-access
    def setUp(self):
        super(TestCLI, self).setUp()
        if not self.fixture_dir.exists():
            self.fixture_dir.mkdir()

        self.tmp_file_path = self.fixture_dir / next(
            tempfile._get_candidate_names())
        while self.tmp_file_path.exists():
            self.tmp_file_path = self.fixture_dir / next(
                tempfile._get_candidate_names())

    def tearDown(self):
        if self.fixture_dir.exists():
            shutil.rmtree(str(self.fixture_dir))

    def test_train(self):
        # Given / When
        train(BEVERAGE_DATASET_PATH, str(self.tmp_file_path), config_path=None,
              verbose=False)

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No trained engine generated")
        msg = "Failed to create an engine from engine dict."
        with self.fail_if_exception(msg):
            SnipsNLUEngine.from_path(self.tmp_file_path)

    def test_parse(self):
        # Given / When
        train(BEVERAGE_DATASET_PATH, str(self.tmp_file_path), config_path=None,
              verbose=False)

        # When
        with self.fail_if_exception("Failed to parse using CLI script"):
            parse(str(self.tmp_file_path), "Make me two cups of coffee")

    def test_cross_val_metrics(self):
        # Given / When
        cross_val_metrics(str(BEVERAGE_DATASET_PATH), str(self.tmp_file_path))

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No metrics found")

    def test_train_test_metrics(self):
        # Given / When
        train_test_metrics(str(BEVERAGE_DATASET_PATH),
                           str(BEVERAGE_DATASET_PATH), str(self.tmp_file_path))

        # Then
        if not self.tmp_file_path.exists():
            self.fail("No metrics found")
