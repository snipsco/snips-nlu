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
          "sklearn-crfsuite",
          "snips"
      ],
      dependency_links=[
          "git+ssh://git@github.com/snipsco/snips-queries-python.git@v0.1.2\
          #egg=snips"
      ],
      packages=[
          "snips_nlu",
          "snips_nlu.nlu_engine"
      ],
      entry_points={},
      include_package_data=False,
      zip_safe=False)
