import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages()
            if "tests" not in p and "debug" not in p]

root = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(root, "snips_nlu", "__about__.py"),
             encoding="utf8") as f:
    about = dict()
    exec(f.read(), about)


with io.open(os.path.join(root, "README.rst"), encoding="utf8") as f:
    readme = f.read()

nlu_metrics_version = "0.12.0"

required = [
    "enum34==1.1.6",
    "pathlib==1.0.1",
    "numpy==1.14.0",
    "scipy==1.0.0",
    "scikit-learn==0.19.1",
    "sklearn-crfsuite==0.3.6",
    "semantic_version==2.6.0",
    "snips_nlu_utils==0.6.1",
    "snips_nlu_ontology==0.57.0",
    "num2words==0.5.6",
    "plac==0.9.6",
    "requests==2.18.4"
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

setup(name=about["__title__"],
      description=about["__summary__"],
      long_description=readme,
      version=about["__version__"],
      author=about["__author__"],
      author_email=about["__email__"],
      license=about["__license__"],
      url=about["__uri__"],
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
              "snips-nlu=snips_nlu.__main__:main"
          ]
      },
      zip_safe=False)
