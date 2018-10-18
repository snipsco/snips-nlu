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

required = [
    "enum34>=1.1,<2.0; python_version<'3.4'",
    "future>=0.16,<0.17",
    "numpy==1.14.0",
    "scipy>=1.0,<2.0",
    "scikit-learn>=0.19,<0.20",
    "sklearn-crfsuite>=0.3.6,<0.4",
    "semantic_version>=2.6,<3.0",
    "snips_nlu_utils>=0.7,<0.8",
    "snips_nlu_ontology>=0.61.1,<0.62",
    "num2words>=0.5.6,<0.6",
    "plac>=0.9.6,<1.0",
    "requests>=2.0,<3.0",
    "pathlib==1.0.1; python_version < '3.4'",
]

extras_require = {
    "doc": [
        "sphinx>=1.8,<1.9",
        "sphinxcontrib-napoleon>=0.6.1,<0.7",
        "sphinx-rtd-theme>=0.2.4,<0.3"
    ],
    "metrics": [
        "snips_nlu_metrics>=0.13,<0.14",
    ],
    "test": [
        "mock>=2.0,<3.0",
        "snips_nlu_metrics>=0.13,<0.14",
        "pylint>=1.8,<2.0",
        "coverage>=4.4.2,<5.0"
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
