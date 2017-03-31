import os

from setuptools import setup, find_packages
from snips_nlu.languages import Language

packages = find_packages()
language_resources = [os.path.join("snips-nlu-resources", l.iso_code, "*") for l
                      in Language]

setup(name="snips_nlu",
      version="0.0.3",
      description="",
      author="Clement Doumouro",
      author_email="clement.doumouro@snips.ai",
      url="",
      download_url="",
      license="MIT",
      install_requires=[
          "duckling==0.0.11",
          "pytest",
          "enum34",
          "mock",
          "numpy==1.12.1",
          "scipy==0.19.0",
          "scikit-learn==0.18.1",
          "sklearn-crfsuite==0.3.5",
          "snips-queries==0.1.0a1"
      ],
      packages=packages,
      package_data={"": language_resources + ["tests/resources/*"]},
      entry_points={},
      include_package_data=True,
      zip_safe=False)
