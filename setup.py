import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages()
            if "tests" not in p and "debug" not in p]

PACKAGE_NAME = "snips_nlu"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(ROOT_PATH, "README.rst")
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)
VERSION = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION)) as f:
    version = f.readline().strip()

nlu_metrics_version = "0.12.0"

required = [
    "enum34==1.1.6",
    "numpy==1.14.0",
    "scipy==1.0.0",
    "scikit-learn==0.19.1",
    "sklearn-crfsuite==0.3.6",
    "semantic_version==2.6.0",
    "snips_nlu_utils==0.6.1",
    "snips_nlu_ontology==0.55.0",
    "num2words==0.5.6",
    "pygments==2.2.0",
]

extras_require = {
    "doc": [
        "sphinx==1.7.1",
        "sphinxcontrib-napoleon==0.6.1",
        "sphinx-rtd-theme==0.2.4"
    ],
    "metrics": [
        "snips_nlu_metrics==%s" % nlu_metrics_version,
    ],
    "test": [
        "mock==2.0.0",
        "snips_nlu_metrics==%s" % nlu_metrics_version,
        "pylint==1.8.2",
        "coverage==4.4.2"
    ],
    "integration_test": [
        "snips_nlu_metrics==%s" % nlu_metrics_version,
    ]
}

with io.open(README, encoding="utf8") as f:
    readme = f.read()

setup(name=PACKAGE_NAME,
      description="Snips Natural Language Understanding library",
      long_description=readme,
      version=version,
      author="Clement Doumouro, Adrien Ball",
      author_email="clement.doumouro@snips.ai, adrien.ball@snips.ai",
      license="Apache 2.0",
      install_requires=required,
      extras_require=extras_require,
      classifiers=[
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
      ],
      packages=packages,
      include_package_data=True,
      entry_points={
          "console_scripts": [
              "train-engine=cli.cli:main_train_engine",
              "engine-inference=cli.cli:main_engine_inference",
              "cross-val-metrics=cli.cli:main_cross_val_metrics",
              "train-test-metrics=cli.cli:main_train_test_metrics",
              "generate-dataset=snips_nlu_dataset:main_generate_dataset"
          ]
      },
      zip_safe=False)
