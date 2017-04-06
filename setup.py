from setuptools import setup, find_packages

packages = find_packages()

setup(name="snips_nlu",
      version="0.2.3",
      description="",
      author="Clement Doumouro",
      author_email="clement.doumouro@snips.ai",
      url="",
      download_url="",
      license="MIT",
      install_requires=[
          "duckling==0.0.13",
          "pytest",
          "enum34",
          "mock",
          "numpy==1.12.1",
          "scipy==0.19.0",
          "scikit-learn==0.18.1",
          "sklearn-crfsuite==0.3.5",
          "snips-queries==0.4.0"
      ],
      packages=packages,
      package_data={
          "": [
              "snips-nlu-resources/en/*",
              "snips-nlu-resources/fr/*",
              "snips-nlu-resources/sp/*",
              "tests/resources/*"
          ]},
      entry_points={},
      include_package_data=True,
      zip_safe=False)
