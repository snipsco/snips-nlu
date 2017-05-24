import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages() if "tests" not in p]

PACKAGE_NAME = "snips_nlu"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)
VERSION = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION)) as f:
    version = f.readline().strip()

required = [
    "pytest",
    "enum34==1.1.6",
    "mock",
    "numpy==1.12.1",
    "scipy==0.19.0",
    "scikit-learn==0.18.1",
    "sklearn-crfsuite==0.3.5",
    "builtin_entities_ontology==0.2.0",
    "semantic_version==2.6.0",
    "rustling==1.2",
]

setup(name=PACKAGE_NAME,
      version=version,
      author="Clement Doumouro",
      author_email="clement.doumouro@snips.ai",
      license="MIT",
      install_requires=required,
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
