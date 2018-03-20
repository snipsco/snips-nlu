from __future__ import unicode_literals

import logging

logger = logging.getLogger("snips_nlu")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler((handler))
