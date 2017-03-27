from setuptools import setup, find_packages

packages = find_packages()

setup(name="snips_nlu",
      version="0.0.1",
      description="",
      author="Clement Doumouro",
      author_email="clement.doumouro@snips.ai",
      url="",
      download_url="",
      license="MIT",
      install_requires=[
          "pytest",
          "enum34",
          "mock",
          "numpy",
          "scipy",
          "scikit-learn",
          "sklearn-crfsuite"
      ],
      packages=packages,
      package_data={"": ["snips-nlu-resources/*"]},
      entry_points={},
      include_package_data=False,
      zip_safe=False)
