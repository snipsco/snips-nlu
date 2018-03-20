from __future__ import unicode_literals

import logging.config
import os

import yaml

from utils import TEST_PATH

logger = logging.getLogger(__name__)

logging_config_path = os.path.join(TEST_PATH, "logging.yaml")
env_key = "LOG_CFG"
value = os.getenv(env_key, None)
if value:
    path = value

with open(logging_config_path, 'rt') as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)
