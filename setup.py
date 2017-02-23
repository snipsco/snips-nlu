from setuptools import setup
from setuptools import find_packages
import os

setup(name='custom_intent_parser',
    version='0.1.0',
    description='',
    author='Clement Doumouro',
    author_email='clement.doumouro@snips.ai',
    url='',
    download_url='',
    license='MIT',
    install_requires=[],
    packages=['custom_intent_parser', 'custom_intent_parser.entity_extractor', 'custom_intent_parser.intent_parser', 'custom_intent_parser.intent_parser.rasa_intent_parser'],
    entry_points={},
    include_package_data=False,
    zip_safe=False)