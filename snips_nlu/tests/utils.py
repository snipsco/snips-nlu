from __future__ import unicode_literals

import json
import shutil
import sys
import tempfile
import traceback as tb
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase

from snips_nlu_ontology import get_all_languages

from snips_nlu.common.utils import json_string, unicode_string
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_parser import IntentParser
from snips_nlu.resources import load_resources
from snips_nlu.result import empty_result
from snips_nlu.slot_filler import SlotFiller

TEST_PATH = Path(__file__).parent
TEST_RESOURCES_PATH = TEST_PATH / "resources"
PERFORMANCE_DATASET_PATH = TEST_RESOURCES_PATH / "performance_dataset.json"


# pylint: disable=invalid-name
class SnipsTest(TestCase):

    def setUp(self):
        for l in get_all_languages():
            load_resources(l)

    @contextmanager
    def fail_if_exception(self, msg):
        try:
            yield
        except Exception:  # pylint: disable=W0703
            trace = tb.format_exc()
            self.fail("{}\b{}".format(msg, trace))

    def assertJsonContent(self, json_path, expected_dict):
        if not json_path.exists():
            self.fail("Json file not found: %s" % str(json_path))
        with json_path.open(encoding="utf8") as f:
            data = json.load(f)
        self.assertDictEqual(expected_dict, data)

    def assertFileContent(self, path, expected_content):
        if not path.exists():
            self.fail("File not found: %s" % str(path))
        with path.open(encoding="utf8") as f:
            data = f.read()
        self.assertEqual(expected_content, data)

    @staticmethod
    def writeJsonContent(path, json_dict):
        json_content = json_string(json_dict)
        with path.open(mode="w") as f:
            f.write(json_content)

    @staticmethod
    def writeFileContent(path, content):
        with path.open(mode="w") as f:
            f.write(unicode_string(content))


class FixtureTest(SnipsTest):
    # pylint: disable=protected-access
    def setUp(self):
        super(FixtureTest, self).setUp()
        self.fixture_dir = Path(tempfile.mkdtemp())
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


def get_empty_dataset(language):
    return {
        "intents": {},
        "entities": {},
        "language": language,
    }


with PERFORMANCE_DATASET_PATH.open(encoding='utf8') as dataset_file:
    PERFORMANCE_DATASET = json.load(dataset_file)


class _RedirectStream(object):
    _stream = None

    def __init__(self, new_target):
        self._new_target = new_target
        # We use a list of old targets to make this CM re-entrant
        self._old_targets = []

    def __enter__(self):
        self._old_targets.append(getattr(sys, self._stream))
        setattr(sys, self._stream, self._new_target)
        return self._new_target

    def __exit__(self, exctype, excinst, exctb):
        setattr(sys, self._stream, self._old_targets.pop())


class redirect_stdout(_RedirectStream):
    """Context manager for temporarily redirecting stdout to another file"""

    _stream = "stdout"


class MockIntentParser(IntentParser):
    def fit(self, dataset, force_retrain):
        self._fitted = True
        return self

    @property
    def fitted(self):
        return hasattr(self, '_fitted') and self._fitted

    def parse(self, text, intents=None, top_n=None):
        return empty_result(text)

    def get_intents(self, text):
        return []

    def get_slots(self, text, intent):
        return []

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        with (path / "metadata.json").open(mode="w") as f:
            f.write(json_string({"unit_name": self.unit_name}))

    @classmethod
    def from_path(cls, path, **shared):
        cfg = cls.config_type()
        return cls(cfg)


class MockIntentClassifier(IntentClassifier):
    _fitted = False

    @property
    def fitted(self):
        return self._fitted

    def fit(self, dataset):
        self._fitted = True
        return self

    def get_intent(self, text, intents_filter):
        return None

    def get_intents(self, text):
        return []

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        with (path / "metadata.json").open(mode="w") as f:
            f.write(json_string({"unit_name": self.unit_name}))

    @classmethod
    def from_path(cls, path, **shared):
        config = cls.config_type()
        return cls(config)


class MockSlotFiller(SlotFiller):
    _fitted = False

    @property
    def fitted(self):
        return self._fitted

    def get_slots(self, text):
        return []

    def fit(self, dataset, intent):
        self._fitted = True
        return self

    def persist(self, path):
        path = Path(path)
        path.mkdir()
        with (path / "metadata.json").open(mode="w") as f:
            f.write(json_string({"unit_name": self.unit_name}))

    @classmethod
    def from_path(cls, path, **shared):
        config = cls.config_type()
        return cls(config)
