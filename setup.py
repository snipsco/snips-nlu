import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages()
            if "tests" not in p and "debug" not in p]

PACKAGE_NAME = "snips_nlu"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)
VERSION = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION)) as f:
    version = f.readline().strip()

required = [
    "enum34==1.1.6",
    "numpy==1.12.1",
    "scipy==0.19.0",
    "scikit-learn==0.18.1",
    "sklearn-crfsuite==0.3.5",
    "builtin_entities_ontology==0.4.2",
    "semantic_version==2.6.0",
    "rustling==7.2",
    "nlu_utils==0.5.0",
    "num2words==0.5.5"
]

extras_require = {
    "test": [
        "mock==2.0.0",
        "nlu_metrics==0.7.4",
        "snips_nlu_rust==0.29.0",
        "pylint==1.7.4"
    ]
}

setup(name=PACKAGE_NAME,
      version=version,
      author="Clement Doumouro, Adrien Ball",
      author_email="clement.doumouro@snips.ai, adrien.ball@snips.ai",
      license="MIT",
      install_requires=required,
      extras_require=extras_require,
      packages=packages,
      package_data={
          "": [
              VERSION,
              "snips-nlu-resources/en/*",
              "snips-nlu-resources/fr/*",
              "snips-nlu-resources/es/*",
              "snips-nlu-resources/de/*",
              "snips-nlu-resources/ko/*",
          ]},
      entry_points={
          "console_scripts": [
              "train-assistant=snips_nlu.nlu_engine:main_create_and_train_engine"
          ]
      },
      classifiers=[
          "Programming Language :: Python :: 2.7"
      ],
      include_package_data=True,
      zip_safe=False)
