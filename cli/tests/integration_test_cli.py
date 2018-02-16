# coding=utf-8
from __future__ import unicode_literals

import io
import json
import os
import shutil
import tempfile

from future.moves import sys
from mock import patch

from cli.cli import main_train_engine, main_cross_val_metrics, \
    main_train_test_metrics
from snips_nlu import SnipsNLUEngine
from snips_nlu.tests.utils import TEST_PATH, BEVERAGE_DATASET_PATH, \
    SnipsTest


def mk_sys_argv(args):
    return ["program_name"] + args


class TestCLI(SnipsTest):
    fixture_dir = os.path.join(TEST_PATH, "cli_fixture")

    # pylint: disable=protected-access
    def setUp(self):
        if not os.path.exists(self.fixture_dir):
            os.mkdir(self.fixture_dir)

        self.tmp_file_path = os.path.join(
            self.fixture_dir, next(tempfile._get_candidate_names()))
        while os.path.exists(self.tmp_file_path):
            self.tmp_file_path = os.path.join(
                self.fixture_dir, next(tempfile._get_candidate_names()))

    def tearDown(self):
        if os.path.exists(self.fixture_dir):
            shutil.rmtree(self.fixture_dir)

    def test_main_train_engine(self):
        # Given
        args = [BEVERAGE_DATASET_PATH, self.tmp_file_path]
        with patch.object(sys, "argv", mk_sys_argv(args)):
            # When
            main_train_engine()

            # Then
            if not os.path.exists(self.tmp_file_path):
                self.fail("No trained engine generated")
            msg = "Failed to create an engine from engine dict."
            with self.fail_if_exception(msg):
                with io.open(self.tmp_file_path, "r", encoding="utf8") as f:
                    trained_engine_dict = json.load(f)
                SnipsNLUEngine.from_dict(trained_engine_dict)

    def test_main_cross_val_metrics(self):
        # Given
        args = [BEVERAGE_DATASET_PATH, self.tmp_file_path]
        with patch.object(sys, "argv", mk_sys_argv(args)):
            # When
            main_cross_val_metrics()

            # Then
            if not os.path.exists(self.tmp_file_path):
                self.fail("No metrics found")

    def test_main_train_test_metrics(self):
        # Given
        args = [BEVERAGE_DATASET_PATH, BEVERAGE_DATASET_PATH,
                self.tmp_file_path]
        with patch.object(sys, "argv", mk_sys_argv(args)):
            # When
            main_train_test_metrics()

            # Then
            if not os.path.exists(self.tmp_file_path):
                self.fail("No metrics found")
