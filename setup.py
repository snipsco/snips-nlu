from setuptools import setup

setup(name="snips_nlu",
      version="0.0.1",
      description="",
      author="Clement Doumouro",
      author_email="clement.doumouro@snips.ai",
      url="",
      download_url="",
      license="MIT",
      install_requires=[
          "enum34",
          "mock",
          "sklearn-crfsuite",
          "snips-queries"
      ],
      packages=[
          "snips_nlu"
      ],
      entry_points={},
      include_package_data=False,
      zip_safe=False)
