# coding=utf-8
from __future__ import unicode_literals

import json
import shutil
import tempfile
from builtins import str

from future.moves import sys
from mock import patch

from snips_nlu import SnipsNLUEngine
from snips_nlu.cli.cli import (
    main_train_engine, main_cross_val_metrics, main_train_test_metrics)
from snips_nlu.tests.utils import (
    TEST_PATH, BEVERAGE_DATASET_PATH, SnipsTest)


def mk_sys_argv(args):
    return ["program_name"] + args


class TestCLI(SnipsTest):
    fixture_dir = TEST_PATH / "cli_fixture"

    # pylint: disable=protected-access
    def setUp(self):
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

    def test_main_train_engine(self):
        # Given
        args = [str(BEVERAGE_DATASET_PATH), str(self.tmp_file_path)]
        with patch.object(sys, "argv", mk_sys_argv(args)):
            # When
            main_train_engine()

            # Then
            if not self.tmp_file_path.exists():
                self.fail("No trained engine generated")
            msg = "Failed to create an engine from engine dict."
            with self.fail_if_exception(msg):
                with self.tmp_file_path.open(mode="r", encoding="utf8") as f:
                    trained_engine_dict = json.load(f)
                SnipsNLUEngine.from_dict(trained_engine_dict)

    def test_main_cross_val_metrics(self):
        # Given
        args = [str(BEVERAGE_DATASET_PATH), str(self.tmp_file_path)]
        with patch.object(sys, "argv", mk_sys_argv(args)):
            # When
            main_cross_val_metrics()

            # Then
            if not self.tmp_file_path.exists():
                self.fail("No metrics found")

    def test_main_train_test_metrics(self):
        # Given
        args = [str(BEVERAGE_DATASET_PATH), str(BEVERAGE_DATASET_PATH),
                str(self.tmp_file_path)]
        with patch.object(sys, "argv", mk_sys_argv(args)):
            # When
            main_train_test_metrics()

            # Then
            if not self.tmp_file_path.exists():
                self.fail("No metrics found")
