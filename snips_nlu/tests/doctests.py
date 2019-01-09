import doctest
import unittest

from future.moves import sys

import snips_nlu.dataset
import snips_nlu.result

doctest_modules = [
    snips_nlu.dataset.entity,
    snips_nlu.dataset.intent,
    snips_nlu.dataset.dataset,
    snips_nlu.result
]

suite = unittest.TestSuite()
for mod in doctest_modules:
    suite.addTest(doctest.DocTestSuite(mod))
runner = unittest.TextTestRunner()

if not runner.run(suite).wasSuccessful():
    sys.exit(1)

