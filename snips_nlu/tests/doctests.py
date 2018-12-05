import doctest
import unittest

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
runner.run(suite)
