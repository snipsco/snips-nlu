from setuptools import setup, find_packages

packages = find_packages()

setup(name="snips_nlu",
      version="0.1.0",
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
      package_data={
          "": [
              "snips-nlu-resources/eng/*",
              "snips-nlu-resources/fra/*",
              "snips-nlu-resources/spa/*",
              "tests/resources/*"
          ]},
      entry_points={},
      include_package_data=True,
      zip_safe=False)
