import io
import os

from setuptools import setup, find_packages

packages = find_packages()

PACKAGE_NAME = "snips_nlu"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)
VERSION = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION)) as f:
    version = f.readline().strip()

required = [
    "duckling==0.0.15",
    "pytest",
    "enum34==1.1.6",
    "mock",
    "numpy==1.12.1",
    "scipy==0.19.0",
    "scikit-learn==0.18.1",
    "sklearn-crfsuite==0.3.5",
    "snips-queries==0.4.0",
    "builtin_entities_ontology==0.1.0"
]

test_required = [
    "semantic_version==2.6.0"
]

setup(name=PACKAGE_NAME,
      version=version,
      author="Clement Doumouro",
      author_email="clement.doumouro@snips.ai",
      license="MIT",
      install_requires=required,
      extras_require={"test": test_required},
      packages=packages,
      package_data={
          "": [
              VERSION,
              "snips-nlu-resources/en/*",
              "snips-nlu-resources/fr/*",
              "snips-nlu-resources/es/*",
              "snips-nlu-resources/de/*",
              "snips-nlu-resources/ko/*",
              "tests/resources/*"
          ]},
      entry_points={},
      include_package_data=True,
      zip_safe=False)
